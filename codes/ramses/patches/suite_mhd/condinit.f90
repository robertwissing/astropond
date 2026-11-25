!================================================================
! condinit.f90 -- testsuite MHD patch (PATCH=../patches/suite_mhd)
!
! FULL replacement of mhd/condinit.f90 (VPATH shadows the whole file): all
! stock condinit_kind cases are kept below; this patch adds a 'gridic' kind
! that initialises hydro + FACE-CENTRED B from a GridIC file (see
! setupfiles/sph_interp/gridic.py): a fine UNIFORM field of
! (rho,vx,vy,vz,P[,Bx_l,Bx_r,By_l,By_r,Bz_l,Bz_r]) deposited from the SPH IC
! (petkova) with the 6 div-free face-B components from project_solenoidal --
! Bx_l at face i-1/2, Bx_r at i+1/2, matching RAMSES's q(6:8)/q(nvar+1:nvar+3)
! convention exactly. Namelist:  &INIT_PARAMS condinit_kind='gridic'
! Reads './ic_gridic.bin' once (first cell). NDIM=3 builds only (SOLVERmhd is
! inherently 3-component). A hydro GridIC (mhd flag 0) runs with B=0.
!================================================================
module gridic_data
  use amr_parameters, only: dp
  implicit none
  integer :: gic_nx=0, gic_ny=0, gic_nz=0, gic_mhd=0
  real(dp) :: gic_boxlen=1.0d0, gic_gamma=1.6666666667d0, gic_time=0.0d0
  real(dp) :: gic_box(3)=1.0d0
  real(dp), allocatable :: gic_rho(:), gic_vx(:), gic_vy(:), gic_vz(:), gic_P(:)
  real(dp), allocatable :: gic_bxl(:), gic_bxr(:), gic_byl(:), gic_byr(:)
  real(dp), allocatable :: gic_bzl(:), gic_bzr(:)
  logical :: gic_loaded=.false.
  integer, parameter :: GIC_MAGIC=1196575043   ! 0x47524943 ('GRIC' bytes on LE disk)
contains
  subroutine gridic_load(filename)
    character(len=*), intent(in) :: filename
    integer(kind=4) :: magic, version, nx, ny, nz, nvar_f, mhd
    real(kind=8) :: hdr(6)
    integer :: n, iu, ierr
    if (gic_loaded) return
    open(newunit=iu, file=trim(filename), form='unformatted', access='stream', &
         status='old', iostat=ierr)
    if (ierr /= 0) then
       write(*,*) '[gridic] FATAL: cannot open ', trim(filename)
       stop
    endif
    read(iu) magic, version, nx, ny, nz, nvar_f
    if (magic /= GIC_MAGIC) then
       write(*,*) '[gridic] FATAL: bad magic in ', trim(filename), magic
       stop
    endif
    read(iu) mhd
    read(iu) hdr                 ! boxlen, gamma, time, box(1:3)
    gic_nx=nx; gic_ny=ny; gic_nz=nz; gic_mhd=mhd
    gic_boxlen=hdr(1); gic_gamma=hdr(2); gic_time=hdr(3); gic_box=hdr(4:6)
    n = nx*ny*nz
    allocate(gic_rho(n), gic_vx(n), gic_vy(n), gic_vz(n), gic_P(n))
    allocate(gic_bxl(n), gic_bxr(n), gic_byl(n), gic_byr(n))
    allocate(gic_bzl(n), gic_bzr(n))
    read(iu) gic_rho
    read(iu) gic_vx
    read(iu) gic_vy
    read(iu) gic_vz
    read(iu) gic_P
    if (mhd /= 0) then
       read(iu) gic_bxl
       read(iu) gic_bxr
       read(iu) gic_byl
       read(iu) gic_byr
       read(iu) gic_bzl
       read(iu) gic_bzr
    else
       gic_bxl=0d0; gic_bxr=0d0; gic_byl=0d0; gic_byr=0d0
       gic_bzl=0d0; gic_bzr=0d0
    endif
    close(iu)
    gic_loaded=.true.
    write(*,'(A,3I5,A,1PE12.4,A,I2)') ' [gridic] loaded ', nx, ny, nz, &
         ' boxlen=', gic_boxlen, ' mhd=', gic_mhd
  end subroutine gridic_load

  ! C-order flat index for 0-based (ix,iy,iz): matches gridic.py (ix slowest).
  pure integer function gic_idx(ix,iy,iz)
    integer, intent(in) :: ix,iy,iz
    gic_idx = (ix*gic_ny + iy)*gic_nz + iz + 1
  end function gic_idx
end module gridic_data


!================================================================
!================================================================
!================================================================
!================================================================
subroutine  condinit(x,u,dx,nn)
  use amr_commons
  use hydro_parameters
  use gridic_data
  implicit none
  integer ::nn                            ! Number of cells
  real(dp)::dx                            ! Cell size
  real(dp),dimension(1:nvector,1:nvar+3)::u ! Conservative variables
  real(dp),dimension(1:nvector,1:ndim)::x ! Cell center position.
  logical,save:: first_call = .true.       ! True if this is the first call to condinit

  !================================================================
  ! This routine generates initial conditions for RAMSES.
  ! Positions are in user units:
  ! x(i,1:3) are in [0,boxlen]**ndim.
  ! U is the conservative variable vector. Conventions are here:
  ! U(i,1): d, U(i,2:4): d.u,d.v,d.w, U(i,5): E, U(i,6:8): Bleft,
  ! U(i,nvar+1:nvar+3): Bright
  ! Q is the primitive variable vector. Conventions are here:
  ! Q(i,1): d, Q(i,2:4):u,v,w, Q(i,5): P, Q(i,6:8): Bleft,
  ! Q(i,nvar+1:nvar+3): Bright
  ! If nvar > 8, remaining variables (9:nvar) are treated as passive
  ! scalars in the hydro solver.
  ! U(:,:) and Q(:,:) are in user units.
  !================================================================
#if NENER>0
  integer::irad
#endif
#if NVAR>NHYDRO+NENER
  integer::ivar
#endif
  real(dp),dimension(1:nvector,1:nvar+3),save::q   ! Primitive variables

  select case (condinit_kind)

  case('gridic')
     ! GridIC: SPH-deposited fine uniform field incl. div-free face B
     if (first_call) call gridic_load('ic_gridic.bin')
     call gridic_condinit(x, q, dx, nn)

  case('region')
    ! Call built-in initial condition generator
     call region_condinit(x, q, dx, nn)

  case('orzag_tang')
     call orzag_tang_condinit(x, q, dx, nn)

  case('ponomarenko')
      call ponomarenko_condinit(x, q, dx, nn)

  case('collapse')
     call collapse_condinit(x, q, dx, nn)

  ! Add here, if you wish, some user-defined initial conditions
  ! ........

  case DEFAULT
     if (myid == 1.and. first_call)  write(*,*) "[condinit] Void or invalid condinit_kind, using default IC"
     call region_condinit(x, q, dx, nn)

  end select

  first_call = .false.

  ! Convert primitive to conservative variables
  ! density -> density
  u(1:nn,1)=q(1:nn,1)
  ! velocity -> momentum
  u(1:nn,2)=q(1:nn,1)*q(1:nn,2)
  u(1:nn,3)=q(1:nn,1)*q(1:nn,3)
  u(1:nn,4)=q(1:nn,1)*q(1:nn,4)
  ! kinetic energy
  u(1:nn,neul)=0.0d0
  u(1:nn,neul)=u(1:nn,neul)+0.5*q(1:nn,1)*q(1:nn,2)**2
  u(1:nn,neul)=u(1:nn,neul)+0.5*q(1:nn,1)*q(1:nn,3)**2
  u(1:nn,neul)=u(1:nn,neul)+0.5*q(1:nn,1)*q(1:nn,4)**2
  ! pressure -> total fluid energy
  u(1:nn,neul)=u(1:nn,neul)+q(1:nn,neul)/(gamma-1.0d0)
  ! magnetic energy -> total fluid energy
  u(1:nn,neul)=u(1:nn,neul)+0.125d0*(q(1:nn,6)+q(1:nn,nvar+1))**2
  u(1:nn,neul)=u(1:nn,neul)+0.125d0*(q(1:nn,7)+q(1:nn,nvar+2))**2
  u(1:nn,neul)=u(1:nn,neul)+0.125d0*(q(1:nn,8)+q(1:nn,nvar+3))**2
  u(1:nn,6:8)=q(1:nn,6:8)
  u(1:nn,nvar+1:nvar+3)=q(1:nn,nvar+1:nvar+3)
#if NENER>0
  ! radiative pressure -> radiative energy
  ! radiative energy -> total fluid energy
  do irad=1,nener
     u(1:nn,nhydro+irad)=q(1:nn,nhydro+irad)/(gamma_rad(irad)-1.0d0)
     u(1:nn,neul)=u(1:nn,neul)+u(1:nn,nhydro+irad)
  enddo
#endif
#if NVAR>NHYDRO+NENER
  ! passive scalars
  do ivar=nhydro+1+nener,nvar
     u(1:nn,ivar)=q(1:nn,1)*q(1:nn,ivar)
  end do
#endif

end subroutine condinit
!================================================================
!================================================================
!================================================================
!================================================================
subroutine orzag_tang_condinit(x,q,dx,nn)
  use amr_parameters
  use hydro_parameters
  implicit none
  integer ::nn                            ! Number of cells
  real(dp)::dx                            ! Cell size
  real(dp),dimension(1:nvector,1:nvar+3)::q ! Primitive variables
  real(dp),dimension(1:nvector,1:ndim)::x ! Cell center position.
  !================================================================
  ! This routine generates Orzag-Tang initial conditions for RAMSES.
  !================================================================
  integer::i
  real(dp)::xc,xr,xl,yl,yr,yc,al,ar,B0,pi

  pi=ACOS(-1.0_dp)
  B0=1.0_dp/sqrt(4.0_dp*pi)
  do i=1,nn

     xl=x(i,1)-0.5_dp*dx
     xr=x(i,1)+0.5_dp*dx
     xc=x(i,1)
     yl=x(i,2)-0.5_dp*dx
     yr=x(i,2)+0.5_dp*dx
     yc=x(i,2)

     q(i,1)=25.0_dp/(36.0_dp*pi)
     q(i,2)=-sin(2.0_dp*pi*yc)
     q(i,3)=+sin(2.0_dp*pi*xc)
     q(i,4)=0.0_dp
     q(i,5)=5.0_dp/(12.0_dp*pi)

     Ar = B0*(cos(4.0_dp*pi*xl)/(4.0_dp*pi)+cos(2.0_dp*pi*yr)/(2.0_dp*pi))
     Al = B0*(cos(4.0_dp*pi*xl)/(4.0_dp*pi)+cos(2.0_dp*pi*yl)/(2.0_dp*pi))
     q(i,6)=(Ar-Al)/dx
     Ar = B0*(cos(4.0_dp*pi*xr)/(4.0_dp*pi)+cos(2.0_dp*pi*yr)/(2.0_dp*pi))
     Al = B0*(cos(4.0_dp*pi*xr)/(4.0_dp*pi)+cos(2.0_dp*pi*yl)/(2.0_dp*pi))
     q(i,nvar+1)=(Ar-Al)/dx
     Ar = B0*(cos(4.0_dp*pi*xr)/(4.0_dp*pi)+cos(2.0_dp*pi*yl)/(2.0_dp*pi))
     Al = B0*(cos(4.0_dp*pi*xl)/(4.0_dp*pi)+cos(2.0_dp*pi*yl)/(2.0_dp*pi))
     q(i,7)=(Al-Ar)/dx
     Ar = B0*(cos(4.0_dp*pi*xr)/(4.0_dp*pi)+cos(2.0_dp*pi*yr)/(2.0_dp*pi))
     Al = B0*(cos(4.0_dp*pi*xl)/(4.0_dp*pi)+cos(2.0_dp*pi*yr)/(2.0_dp*pi))
     q(i,nvar+2)=(Al-Ar)/dx

     q(i,8)=0.0_dp
     q(i,nvar+3)=0.0_dp
  end do

end subroutine orzag_tang_condinit
!================================================================
!================================================================
!================================================================
!================================================================
subroutine collapse_condinit(x,q,dx,nn)
  use amr_commons, only:myid
  use amr_parameters
  use hydro_commons
  use poisson_parameters
  use constants, only:mH,kB,M_sun,pc2cm
  implicit none
  integer ::nn                              ! Number of cells
  real(dp)::dx                              ! Cell size
  real(dp),dimension(1:nvector,1:nvar+3)::q ! Primitive variables
  real(dp),dimension(1:nvector,1:ndim)::x ! Cell center position.
  !================================================================
  ! This routine generates initial conditions of a collapsing core
  !================================================================
  integer :: i,j,k,id,iu,iv,iw,ip
  real(dp):: x0,y0,z0,rc,rs,xx,yy,zz,pi,r0,d0,B0,p0,omega0,mass_c_cu,scale_m
  integer :: ivar, np
  real(dp)::scale_nH,scale_T2,scale_l,scale_d,scale_t,scale_v
  real(dp),dimension(1:3,1:3):: rot_M,rot_invM,rot_tilde
  real(dp):: theta_mag_radians

  logical,save:: first=.true.
  real(dp),dimension(1:3,1:100,1:100,1:100),save::q_idl
  real(dp),save::vx_tot,vy_tot,vz_tot,vx2_tot,vy2_tot,vz2_tot
  integer,save:: n_size
  integer:: ind_i, ind_j, ind_k
  real(dp),save:: ind,seed1,seed2,seed3,xi,yi,zi,vx,vy,vz
  real(dp),save:: C_s,v_rms
  integer, save :: count_vrms

  id=1; iu=2; iv=3; iw=4; ip=5
  x0=0.5*boxlen
  y0=0.5*boxlen
  z0=0.5*boxlen
  pi=acos(-1.0d0)

  ! Conversion factor from user units to cgs units
  call units(scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)
  scale_m=scale_d*scale_l**ndim

  ! cloud mass (warning mass_c should not be changed, because condinit called not just once)
  ! mass_c    is in solar mass
  ! mass_c_cu is in code units
  mass_c_cu = mass_c * (M_sun / scale_m )

  ! cloud radius
  r0=(alpha_dense_core*2.*6.67d-8*mass_c_cu*scale_m*mu_gas*mH/(5.*kB*T_eos))/scale_l
  ! cloud density
  d0 = 3.0d0*mass_c_cu/(4.0d0*pi*r0**3.)

  ! cloud rotation ! remember that G=1 in code units
  omega0 = sqrt(beta_dense_core*4.*pi*d0)

  ! cloud pressure ! remember that G=1 in code units
  p0 = alpha_dense_core*d0*d0*r0*r0*8.*pi/15.

  ! vertical magnetic field ! remember that G=1 in code units (and that B as a factor 1/sqrt(4pi) between SI and Gaussian units)
  B0 = sqrt(4.*pi/5.)/0.53*crit_dense_core*d0*r0
  ! B0 could be defined equivalently as
  !B0 = mass_c_cu*3./sqrt(5.)/0.53*crit_dense_core/r0**2/sqrt(4.*pi)

  ! angle between the rotation axis and the magnetic field
  theta_mag_radians= theta_mag/180.0d0*pi

  rot_M(1,1:3) = (/cos(theta_mag_radians),0.0d0,-sin(theta_mag_radians)/)
  rot_M(2,1:3) = (/0.0d0,1.0d0,0.0d0/)
  rot_M(3,1:3) = (/sin(theta_mag_radians),0.0d0,cos(theta_mag_radians)/)

  rot_invM(1,1:3) = (/cos(theta_mag_radians),0.0d0,sin(theta_mag_radians)/)
  rot_invM(2,1:3) = (/0.0d0,1.0d0,0.0d0/)
  rot_invM(3,1:3) = (/-sin(theta_mag_radians),0.0d0,cos(theta_mag_radians)/)

  rot_tilde(1,1:3) = (/0.0d0,1.0d0,0.0d0/)
  rot_tilde(2,1:3) = (/-1.0d0,0.0d0,0.0d0/)
  rot_tilde(3,1:3) = (/0.0d0,0.0d0,0.0d0/)

  if(first) then
    ! sound speed
    C_s = sqrt(kB*T_eos/(mu_gas*mH))/scale_v
    !C_s could be defined equivalently as sqrt( T_eos / (mu_gas*scale_T2) )

    vx_tot=0.d0
    vy_tot=0.d0
    vz_tot=0.d0
    vx2_tot=0.d0
    vy2_tot=0.d0
    vz2_tot=0.d0
    v_rms=0.d0
    count_vrms=0
    if(Mach .ne. 0)then
      if (myid==1) write(*,*) 'Read the file which contains the initial turbulent velocity field'
      open(20,file='init_turb.data',form='formatted')
      read(20,*) n_size, ind, seed1,seed2,seed3
      if(n_size .ne. 100) then
          write(*,*) 'Unexpected field size'
          stop
      endif
      do k=1,n_size
        do j=1,n_size
          do i=1,n_size
            read(20,*)xi,yi,zi,vx,vy,vz
            q_idl(1,i,j,k) = vx
            q_idl(2,i,j,k) = vy
            q_idl(3,i,j,k) = vz
            xi = boxlen*((i-0.5)/n_size)-x0
            yi = boxlen*((j-0.5)/n_size)-y0
            zi = boxlen*((k-0.5)/n_size)-z0
            rs=sqrt(xi**2+yi**2+zi**2)

            IF(rs .le. r0) THEN
              !print*, vx_tot,vy_tot,vz_tot,vx2_tot,vy2_tot,vz2_tot
              vx_tot = vx_tot + vx
              vy_tot = vy_tot + vy
              vz_tot = vz_tot + vz

              vx2_tot = vx2_tot + vx**2
              vy2_tot = vy2_tot + vy**2
              vz2_tot = vz2_tot + vz**2

              count_vrms=count_vrms+1
            end if
          end do
        end do
      end do
      close(20)
      v_rms=sqrt((vx2_tot+vy2_tot+vz2_tot)/dble(count_vrms)-((vx_tot+vy_tot+vz_tot)/dble(count_vrms))**2)
      if (myid == 1) print *, 'v_rms for given seed =',v_rms
      ! correction factor to have the expected Mach number stored in v_rms
      v_rms = Mach*C_s/v_rms
      if (myid == 1) print *, 'correction factor for turbulent field =',v_rms
   end if

   if(myid==1)then
      print*,'alpha_dense_core=',alpha_dense_core
      print*,'beta_dense_core=',beta_dense_core
      print*,'Mass=',mass_c,' Msun'
      print*,'d0 (in g/cc)=',d0*scale_d
      print*,'Turbulent Mach,cs (km/s)=',Mach,C_s*scale_v/1e5
      print*,'r0,boxlen (in code units)=',r0,boxlen
      print*,'r0,boxlen (in pc)=',r0*scale_l/pc2cm,boxlen*scale_l/pc2cm
    endif
    first = .false.
  end if



  DO i=1,nn
     xx=x(i,1)-x0
     yy=x(i,2)-y0
     zz=x(i,3)-z0
     rc=sqrt(xx**2+yy**2)
     rs=sqrt(xx**2+yy**2+zz**2)

     !Bx component
     q(i,6     ) = 0.
     q(i,nvar+1) = 0.

     !By component
     q(i,7     ) = 0.
     q(i,nvar+2) = 0.

     !Bz component
     q(i,8     ) = B0
     q(i,nvar+3) = B0

     q(i,iu) = 0.
     q(i,iv) = 0.
     q(i,iw) = 0.
     if(Mach .ne. 0)then
      !initialise the turbulent velocity field
      !make a zero order interpolation (should be improved)
      ind_i = int((x(i,1)/boxlen)*n_size)+1
      ind_j = int((x(i,2)/boxlen)*n_size)+1
      ind_k = int((x(i,3)/boxlen)*n_size)+1
      ! safe check
      if( ind_i .lt. 1 .or. ind_i .gt. n_size) write(*,*) 'ind_i ',ind_i,(x(i,1)/boxlen)*n_size+1,n_size
      if( ind_j .lt. 1 .or. ind_j .gt. n_size) write(*,*) 'ind_j ',ind_j
      if( ind_k .lt. 1 .or. ind_k .gt. n_size) write(*,*) 'ind_k ',ind_k
    end if

     IF(rs .le. r0) THEN
        ! if(theta_mag.eq.0.0d0) then
        !   q(i,id) = d0*(1.0+delta_rho*cos(2.*atan(yy/xx)))!(2.0*(xx/rc)**2-1.0))
        !   q(i,iu) = omega0 * yy
        !   q(i,iv) = -omega0 * xx
        !   q(i,iw) = 0.0
        ! else
          q(i,id) = d0*(1.0+delta_rho*cos(2.*atan(yy/(cos(theta_mag_radians)*xx-sin(theta_mag_radians)*zz))))

          if(Mach .ne. 0)then
            !print*,omega0*yy,omega0*xx,v_rms,v_rms*(q_idl(1,ind_i,ind_j,ind_k)-vx_tot/dble(count_vrms))
            q(i,iu) =  v_rms*(q_idl(1,ind_i,ind_j,ind_k)-vx_tot/dble(count_vrms))
            q(i,iv) =  v_rms*(q_idl(2,ind_i,ind_j,ind_k)-vy_tot/dble(count_vrms))
            q(i,iw) =  v_rms*(q_idl(3,ind_i,ind_j,ind_k)-vz_tot/dble(count_vrms))
          end if
          q(i,iu:iw) = q(i,iu:iw) + matmul(rot_invM,omega0*matmul(rot_tilde,matmul(rot_M,(/xx,yy,zz/))))
        ! endif
       q(i,ip) = p0
     ELSE
       q(i,id) = d0/100.
       xx = r0 * xx / rc
       yy = r0 * yy / rc

       if(Mach .ne. 0)then
        q(i,iu) = v_rms*(q_idl(1,ind_i,ind_j,ind_k)-vx_tot/dble(count_vrms))! omega0 * yy
        q(i,iv) = v_rms*(q_idl(2,ind_i,ind_j,ind_k)-vy_tot/dble(count_vrms))!-omega0 * xx
        q(i,iw) = v_rms*(q_idl(3,ind_i,ind_j,ind_k)-vz_tot/dble(count_vrms))
       end if

      !  q(i,iu) = 0.0! omega0 * yy
      !  q(i,iv) = 0.0!-omega0 * xx
      !  q(i,iw) = 0.0

       q(i,ip) = p0/100.
     ENDIF
  ENDDO

end subroutine collapse_condinit
!================================================================
!================================================================
!================================================================
!================================================================
subroutine velana(x,v,dx,t,ncell)
  use amr_parameters
  use hydro_parameters
  implicit none
  integer ::ncell                         ! Size of input arrays
  real(dp)::dx                            ! Cell size
  real(dp)::t                             ! Current time
  real(dp),dimension(1:nvector,1:3)::v    ! Velocity field
  real(dp),dimension(1:nvector,1:ndim)::x ! Cell center position.
  !================================================================
  ! This routine computes the user defined velocity fields.
  ! x(i,1:ndim) are cell center position in [0,boxlen] (user units).
  ! v(i,1:3) is the imposed 3-velocity in user units.
  !================================================================
  integer::i
  real(dp)::xx,yy=0.,zz=0.,vx,vy,vz,aa,twopi
!!$  real(dp)::rr,tt,omega

  ! Add here, if you wish, some user-defined initial conditions
  aa=1.0+0.*t
  twopi=2d0*ACOS(-1d0)
  do i=1,ncell

     xx=x(i,1)+0.*dx
#if NDIM > 1
     yy=x(i,2)
#endif
#if NDIM > 2
     zz=x(i,3)
#endif
     ! ABC
     vx=aa*(cos(twopi*yy)+sin(twopi*zz))
     vy=aa*(sin(twopi*xx)+cos(twopi*zz))
     vz=aa*(cos(twopi*xx)+sin(twopi*yy))

!!$     ! 1D advection test
!!$     vx=1.0_dp
!!$     vy=0.0_dp
!!$     vz=0.0_dp

!!$     ! Ponomarenko
!!$     xx=xx-boxlen/2.0
!!$     yy=yy-boxlen/2.0
!!$     rr=sqrt(xx**2+yy**2)
!!$     if(yy>0)then
!!$        tt=acos(xx/rr)
!!$     else
!!$        tt=-acos(xx/rr)+twopi
!!$     endif
!!$     if(rr<1.0)then
!!$        omega=0.609711
!!$        vz=0.792624
!!$     else
!!$        omega=0.0
!!$        vz=0.0
!!$     endif
!!$     vx=-sin(tt)*rr*omega
!!$     vy=+cos(tt)*rr*omega

     v(i,1)=vx
#if NDIM > 1
     v(i,2)=vy
#endif
#if NDIM > 2
     v(i,3)=vz
#endif
  end do


end subroutine velana
!================================================================
!================================================================
!================================================================
!================================================================
subroutine ponomarenko_condinit(x,u,dx,nn)
  use amr_parameters
  use hydro_parameters
  implicit none
  integer ::nn                            ! Number of cells
  real(dp)::dx                            ! Cell size
  real(dp),dimension(1:nvector,1:nvar+3)::u ! Conservative variables
  real(dp),dimension(1:nvector,1:ndim)::x ! Cell center position.
  !================================================================
  ! This routine generates initial conditions for RAMSES.
  ! Positions are in user units:
  ! x(i,1:3) are in [0,boxlen]**ndim.
  ! U is the conservative variable vector. Conventions are here:
  ! U(i,1): d, U(i,2:4): d.u,d.v,d.w, U(i,5): E, U(i,6:8): Bleft,
  ! U(i,nvar+1:nvar+3): Bright
  ! Q is the primitive variable vector. Conventions are here:
  ! Q(i,1): d, Q(i,2:4):u,v,w, Q(i,5): P, Q(i,6:8): Bleft,
  ! Q(i,nvar+1:nvar+3): Bright
  ! If nvar > 8, remaining variables (9:nvar) are treated as passive
  ! scalars in the hydro solver.
  ! U(:,:) and Q(:,:) are in user units.
  !================================================================
  integer::ivar,i
  real(dp),dimension(1:nvector,1:nvar+3),save::q   ! Primitive variables
  real(dp)::xc,xr,xl,yl,yr,yc,al,ar,r0,a0,twopi,rr,ss,tt

  ! density
  q(1:nn,1)=1.0

  ! velocity
  q(1:nn,2:4)=0.0
  R0=1.0
  do i=1,nn
     xc=x(i,1)-boxlen/2.0
     yc=x(i,2)-boxlen/2.0
     rr = sqrt(xc**2+yc**2)
     if(rr < R0)q(i,4)=1.0
  end do

  ! pressure
  q(1:nn,5)=1.0

  ! magnetic field
  call mag_screw(x,q,dx,nn)

  ! Convert primitive to conservative variables
  ! density -> density
  u(1:nn,1)=q(1:nn,1)
  ! velocity -> momentum
  u(1:nn,2)=q(1:nn,1)*q(1:nn,2)
  u(1:nn,3)=q(1:nn,1)*q(1:nn,3)
  u(1:nn,4)=q(1:nn,1)*q(1:nn,4)
  ! kinetic energy
  u(1:nn,5)=0.0d0
  u(1:nn,5)=u(1:nn,5)+0.5*q(1:nn,1)*q(1:nn,2)**2
  u(1:nn,5)=u(1:nn,5)+0.5*q(1:nn,1)*q(1:nn,3)**2
  u(1:nn,5)=u(1:nn,5)+0.5*q(1:nn,1)*q(1:nn,4)**2
  ! pressure -> total fluid energy
  u(1:nn,5)=u(1:nn,5)+q(1:nn,5)/(gamma-1.0d0)
  ! magnetic energy -> total fluid energy
  u(1:nn,5)=u(1:nn,5)+0.125d0*(q(1:nn,6)+q(1:nn,nvar+1))**2
  u(1:nn,5)=u(1:nn,5)+0.125d0*(q(1:nn,7)+q(1:nn,nvar+2))**2
  u(1:nn,5)=u(1:nn,5)+0.125d0*(q(1:nn,8)+q(1:nn,nvar+3))**2
  u(1:nn,6:8)=q(1:nn,6:8)
  u(1:nn,nvar+1:nvar+3)=q(1:nn,nvar+1:nvar+3)
  ! passive scalars
#if NDIM > 8
  do ivar=9,nvar
     u(1:nn,ivar)=q(1:nn,1)*q(1:nn,ivar)
  end do
#endif

end subroutine ponomarenko_condinit
!================================================================
!================================================================
!================================================================
!================================================================
subroutine mag_screw(x,q,dx,nn)
  use amr_parameters
  use hydro_parameters, only: nvar
  use const
  implicit none

  real(dp),dimension(1:nvector,1:ndim)::x    ! Cell center position
  real(dp),dimension(1:nvector,1:nvar+3)::q  ! Primitive variables
  integer::nn                                ! Number of cells
  real(dp)::dx                               ! Cell size
  real(dp)::xx,yy,zz,A_screw
  integer::i,it,nticks
  real(dp)::dxmin,zfl,zceil,rtick,x_edge,y_edge
  real(dp),dimension(1:2,1:2,1:2)::A_mag     ! x/y,left/right,up/down
  real(dp)::dmax,dnfw

  ! Ponomarenko
  dxmin=boxlen*0.5d0**nlevelmax
  nticks=nint(dx/dxmin)

  do i=1,nn
    ! box-centered coordinates
    xx=x(i,1) - boxlen * 0.5d0
    yy=x(i,2) - boxlen * 0.5d0
    zz=x(i,3) - boxlen * 0.5d0

    zfl   = zz - 0.5d0*dx
    zceil = zz + 0.5d0*dx

    A_mag = 0.0

    ! A_(x,l)
    x_edge = xx + 0.5*(dxmin-dx)
    y_edge = yy - 0.5*dx
    do it=1,nticks
      ! floor
      A_mag(1,1,1) = A_mag(1,1,1) + A_screw(x_edge,y_edge,zfl,1)
      ! ceiling
      A_mag(1,1,2) = A_mag(1,1,2) + A_screw(x_edge,y_edge,zceil,1)
      ! advance along x
      x_edge=x_edge + dxmin
    end do

    ! A_(x,r)
    x_edge = xx + 0.5*(dxmin-dx)
    y_edge = yy + 0.5*dx
    do it=1,nticks
      ! floor
      A_mag(1,2,1) = A_mag(1,2,1) + A_screw(x_edge,y_edge,zfl,1)
      ! ceiling
      A_mag(1,2,2) = A_mag(1,2,2) + A_screw(x_edge,y_edge,zceil,1)
      ! advance along x
      x_edge=x_edge + dxmin
    end do

    ! A_(y,l)
    x_edge = xx - 0.5*dx
    y_edge = yy + 0.5*(dxmin-dx)
    do it=1,nticks
      ! floor
      A_mag(2,1,1) = A_mag(2,1,1) + A_screw(x_edge,y_edge,zfl,2)
      ! ceiling
      A_mag(2,1,2) = A_mag(2,1,2) + A_screw(x_edge,y_edge,zceil,2)
      ! advance along y
      y_edge=y_edge + dxmin
    end do

    ! A_(y,r)
    x_edge = xx + 0.5*dx
    y_edge = yy + 0.5*(dxmin-dx)
    do it=1,nticks
      ! floor
      A_mag(2,2,1) = A_mag(2,2,1) + A_screw(x_edge,y_edge,zfl,2)
      ! ceiling
      A_mag(2,2,2) = A_mag(2,2,2) + A_screw(x_edge,y_edge,zceil,2)
      ! advance along y
      y_edge=y_edge + dxmin
    end do

    ! average value
    A_mag = A_mag / DBLE(nticks)

    ! B left
    ! X direction
    q(i,6)= - (A_mag(2,1,2)-A_mag(2,1,1)) / dx ! [-d/dz A_y]
    ! Y direction
    q(i,7)=   (A_mag(1,1,2)-A_mag(1,1,1)) / dx ! [ d/dz A_x]
    ! Z direction  [d/dx A_y - d/dy A_x]
    q(i,8)=   ( A_mag(2,2,1)-A_mag(2,1,1) - A_mag(1,2,1)+A_mag(1,1,1) ) / dx

    ! B right
    ! X direction
    q(i,nvar+1)= - (A_mag(2,2,2)-A_mag(2,2,1)) / dx ! [-d/dz A_y]
    ! Y direction
    q(i,nvar+2)=   (A_mag(1,2,2)-A_mag(1,2,1)) / dx ! [ d/dz A_x]
    ! Z direction  [d/dx A_y - d/dy A_x]
    q(i,nvar+3)=   ( A_mag(2,2,2)-A_mag(2,1,2) - A_mag(1,2,2)+A_mag(1,1,2) ) / dx
  end do

end subroutine mag_screw
!================================================================
!================================================================
!================================================================
!================================================================
function A_screw(xx,yy,zz,dir)
  use amr_parameters
  use hydro_parameters
  use const
  implicit none
  real(kind=8)::xx,yy,zz,A_screw
  real(kind=8)::A,rr,tt,twopi
  integer::dir

  twopi=2d0*acos(-1d0)

  rr=sqrt(xx**2+yy**2)
  if(yy>0)then
     tt=acos(xx/rr)
  else
     tt=-acos(xx/rr)+twopi
  endif
  A_screw=0d0
  if(rr<2.0)then
     A_screw=1d-1*(2d0-rr)*rr*cos(tt)*sin(twopi*(zz+boxlen/2)/boxlen)
     if(dir==1)A_screw=A_screw*cos(tt)
     if(dir==2)A_screw=A_screw*sin(tt)
  endif

  return
end function A_screw
!================================================================
!================================================================
!================================================================
!================================================================
subroutine velana_ponomarenko(x,v,dx,t,ncell)
  use amr_parameters
  use hydro_parameters
  implicit none
  integer ::ncell                         ! Size of input arrays
  real(dp)::dx                            ! Cell size
  real(dp)::t                             ! Current time
  real(dp),dimension(1:nvector,1:3)::v    ! Velocity field
  real(dp),dimension(1:nvector,1:ndim)::x ! Cell center position.
  !================================================================
  ! This routine computes the user defined velocity fields.
  ! x(i,1:ndim) are cell center position in [0,boxlen] (user units).
  ! v(i,1:3) is the imposed 3-velocity in user units.
  !================================================================
  integer::i
  real(dp)::xx,yy,zz,vx,vy,vz,rr,tt,omega,aa,twopi

  ! Add here, if you wish, some user-defined initial conditions
  aa=1.0
  twopi=2d0*ACOS(-1d0)
  do i=1,ncell

     xx=x(i,1)
#if NDIM > 1
     yy=x(i,2)
#endif
#if NDIM > 2
     zz=x(i,3)
#endif
     ! Ponomarenko
     xx=xx-boxlen/2.0
     yy=yy-boxlen/2.0
     rr=sqrt(xx**2+yy**2)
     if(yy>0)then
        tt=acos(xx/rr)
     else
        tt=-acos(xx/rr)+twopi
     endif
     if(rr<1.0)then
        omega=0.609711
        vz=0.792624
     else
        omega=0.0
        vz=0.0
     endif
     vx=-sin(tt)*rr*omega
     vy=+cos(tt)*rr*omega

     v(i,1)=vx
#if NDIM > 1
     v(i,2)=vy
#endif
#if NDIM > 2
     v(i,3)=vz
#endif
  end do


end subroutine velana_ponomarenko

!================================================================
! Fill PRIMITIVE q from the GridIC fine uniform field. Hydro vars
! (rho, v, P): conservative VOLUME-average of the covered fine cells
! (uniform run: dx == fine dx -> single-cell lookup). Face B: FACE-PLANE
! averages -- B*_l over the low-index face plane of the covered block,
! B*_r over the high-index plane -- NOT volume averages: the fine face
! fluxes telescope, so a coarse cell built this way has divB = 0 exactly
! whenever the fine field does. Periodic wrap on the fine indices.
! NB: no local names colliding with use-associated kind constants
! (amr_parameters has `sp`; Fortran is case-insensitive) -> sum_* prefixes.
!================================================================
subroutine gridic_condinit(x,q,dx,nn)
  use amr_parameters
  use hydro_parameters
  use gridic_data
  implicit none
  integer ::nn
  real(dp)::dx
  real(dp),dimension(1:nvector,1:nvar+3)::q
  real(dp),dimension(1:nvector,1:ndim)::x
  integer::i,ix,iy,iz,ix0,ix1,iy0,iy1,iz0,iz1,idx,ncov,nfa
  real(dp)::finedx,half,sum_r,sum_vx,sum_vy,sum_vz,sum_p
  real(dp)::sum_bxl,sum_bxr,sum_byl,sum_byr,sum_bzl,sum_bzr

  finedx = gic_boxlen/dble(gic_nx)
  half = 0.5d0*dx
  do i=1,nn
     ix0=floor((x(i,1)-half)/finedx); ix1=ceiling((x(i,1)+half)/finedx)-1
     iy0=floor((x(i,2)-half)/finedx); iy1=ceiling((x(i,2)+half)/finedx)-1
     iz0=floor((x(i,3)-half)/finedx); iz1=ceiling((x(i,3)+half)/finedx)-1
     sum_r=0d0; sum_vx=0d0; sum_vy=0d0; sum_vz=0d0; sum_p=0d0; ncov=0
     do ix=ix0,ix1
        do iy=iy0,iy1
           do iz=iz0,iz1
              idx=gic_idx(modulo(ix,gic_nx),modulo(iy,gic_ny),modulo(iz,gic_nz))
              sum_r=sum_r+gic_rho(idx); sum_vx=sum_vx+gic_vx(idx)
              sum_vy=sum_vy+gic_vy(idx); sum_vz=sum_vz+gic_vz(idx)
              sum_p=sum_p+gic_P(idx)
              ncov=ncov+1
           enddo
        enddo
     enddo
     q(i,1)=sum_r/dble(ncov)
     q(i,2)=sum_vx/dble(ncov)
     q(i,3)=sum_vy/dble(ncov)
     q(i,4)=sum_vz/dble(ncov)
     q(i,neul)=sum_p/dble(ncov)
     ! x faces: left = ix0 plane, right = ix1 plane
     sum_bxl=0d0; sum_bxr=0d0; nfa=0
     do iy=iy0,iy1
        do iz=iz0,iz1
           sum_bxl=sum_bxl+gic_bxl(gic_idx(modulo(ix0,gic_nx),modulo(iy,gic_ny),modulo(iz,gic_nz)))
           sum_bxr=sum_bxr+gic_bxr(gic_idx(modulo(ix1,gic_nx),modulo(iy,gic_ny),modulo(iz,gic_nz)))
           nfa=nfa+1
        enddo
     enddo
     q(i,6)=sum_bxl/dble(nfa); q(i,nvar+1)=sum_bxr/dble(nfa)
     ! y faces
     sum_byl=0d0; sum_byr=0d0; nfa=0
     do ix=ix0,ix1
        do iz=iz0,iz1
           sum_byl=sum_byl+gic_byl(gic_idx(modulo(ix,gic_nx),modulo(iy0,gic_ny),modulo(iz,gic_nz)))
           sum_byr=sum_byr+gic_byr(gic_idx(modulo(ix,gic_nx),modulo(iy1,gic_ny),modulo(iz,gic_nz)))
           nfa=nfa+1
        enddo
     enddo
     q(i,7)=sum_byl/dble(nfa); q(i,nvar+2)=sum_byr/dble(nfa)
     ! z faces
     sum_bzl=0d0; sum_bzr=0d0; nfa=0
     do ix=ix0,ix1
        do iy=iy0,iy1
           sum_bzl=sum_bzl+gic_bzl(gic_idx(modulo(ix,gic_nx),modulo(iy,gic_ny),modulo(iz0,gic_nz)))
           sum_bzr=sum_bzr+gic_bzr(gic_idx(modulo(ix,gic_nx),modulo(iy,gic_ny),modulo(iz1,gic_nz)))
           nfa=nfa+1
        enddo
     enddo
     q(i,8)=sum_bzl/dble(nfa); q(i,nvar+3)=sum_bzr/dble(nfa)
  enddo
end subroutine gridic_condinit
