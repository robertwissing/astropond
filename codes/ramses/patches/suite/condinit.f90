!================================================================
! condinit.f90 -- testsuite patch (PATCH=../patch/suite)
!
! Adds a 'gridic' condinit_kind that initialises the hydro from a GridIC file
! (see setupfiles/sph_interp/gridic.py): a fine UNIFORM grid of (rho,vx,vy,vz,P)
! deposited from the SPH IC with our petkova interpolation. Each RAMSES cell at
! (x,dx) -- at ANY level -- is filled with the conservative volume-average of the
! fine GridIC cells it covers, so the SAME file serves a uniform run
! (levelmin=levelmax -> one fine cell per cell) and an AMR run (coarse cells
! average the fine cells). Set in the namelist:  &INIT_PARAMS condinit_kind='gridic'
! The file is read once (first cell) from './ic_gridic.bin' in the run directory.
!
! Hydro only here; MHD (face B) is a separate mhd/condinit.f90 patch.
!================================================================
module gridic_data
  use amr_parameters, only: dp
  implicit none
  integer :: gic_nx=0, gic_ny=0, gic_nz=0, gic_mhd=0
  real(dp) :: gic_boxlen=1.0d0, gic_gamma=1.6666666667d0, gic_time=0.0d0
  real(dp) :: gic_box(3)=1.0d0
  real(dp), allocatable :: gic_rho(:), gic_vx(:), gic_vy(:), gic_vz(:), gic_P(:)
  logical :: gic_loaded=.false.
  integer, parameter :: GIC_MAGIC=1196575043   ! 0x47524943 ('GRIC' bytes on LE disk)
contains
  subroutine gridic_load(filename)
    character(len=*), intent(in) :: filename
    integer(kind=4) :: magic, version, nx, ny, nz, nvar, mhd
    real(kind=8) :: hdr(6)
    integer :: n, iu, ierr
    if (gic_loaded) return
    open(newunit=iu, file=trim(filename), form='unformatted', access='stream', &
         status='old', iostat=ierr)
    if (ierr /= 0) then
       write(*,*) '[gridic] FATAL: cannot open ', trim(filename)
       stop
    endif
    read(iu) magic, version, nx, ny, nz, nvar
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
    read(iu) gic_rho
    read(iu) gic_vx
    read(iu) gic_vy
    read(iu) gic_vz
    read(iu) gic_P
    close(iu)
    gic_loaded=.true.
    write(*,'(A,3I5,A,1PE12.4)') ' [gridic] loaded ', nx, ny, nz, &
         ' boxlen=', gic_boxlen
  end subroutine gridic_load

  ! C-order flat index for 0-based (ix,iy,iz): matches gridic.py (ix slowest).
  pure integer function gic_idx(ix,iy,iz)
    integer, intent(in) :: ix,iy,iz
    gic_idx = (ix*gic_ny + iy)*gic_nz + iz + 1
  end function gic_idx
end module gridic_data


subroutine condinit(x,u,dx,nn)
  use amr_commons
  use hydro_parameters
  use gridic_data
  implicit none
  integer ::nn
  real(dp)::dx
  real(dp),dimension(1:nvector,1:nvar)::u
  real(dp),dimension(1:nvector,1:ndim)::x
  logical,save:: first_call = .true.
#if NENER>0 || NVAR>NHYDRO+NENER
  integer::ivar
#endif
  real(dp),dimension(1:nvector,1:nvar),save::q

  select case (condinit_kind)
  case('gridic')
     if (first_call) call gridic_load('ic_gridic.bin')
     call gridic_condinit(x, q, dx, nn)
  case('region')
     call region_condinit(x, q, dx, nn)
  case('ana_disk_potential')
     call ana_disk_potential_condinit(x, q, dx, nn)
  case DEFAULT
     if (myid==1 .and. first_call) write(*,*) &
          "[condinit] Void or invalid condinit_kind, using default IC"
     call region_condinit(x, q, dx, nn)
  end select

  first_call = .false.

  ! Convert primitive to conservative variables (unchanged from stock RAMSES).
  u(1:nn,1)=q(1:nn,1)
  u(1:nn,2)=q(1:nn,1)*q(1:nn,2)
#if NDIM>1
  u(1:nn,3)=q(1:nn,1)*q(1:nn,3)
#endif
#if NDIM>2
  u(1:nn,4)=q(1:nn,1)*q(1:nn,4)
#endif
  u(1:nn,neul)=0.0d0
  u(1:nn,neul)=u(1:nn,neul)+0.5d0*q(1:nn,1)*q(1:nn,2)**2
#if NDIM>1
  u(1:nn,neul)=u(1:nn,neul)+0.5d0*q(1:nn,1)*q(1:nn,3)**2
#endif
#if NDIM>2
  u(1:nn,neul)=u(1:nn,neul)+0.5d0*q(1:nn,1)*q(1:nn,4)**2
#endif
  u(1:nn,neul)=u(1:nn,neul)+q(1:nn,neul)/(gamma-1.0d0)
#if NENER>0
  do ivar=1,nener
     u(1:nn,nhydro+ivar)=q(1:nn,nhydro+ivar)/(gamma_rad(ivar)-1.0d0)
     u(1:nn,neul)=u(1:nn,neul)+u(1:nn,nhydro+ivar)
  enddo
#endif
#if NVAR>NHYDRO+NENER
  do ivar=nhydro+1+nener,nvar
     u(1:nn,ivar)=q(1:nn,1)*q(1:nn,ivar)
  end do
#endif
end subroutine condinit


!================================================================
! Fill PRIMITIVE q (rho, u,v,w, P) from the GridIC fine uniform field by
! conservative volume-average of the fine cells covered by each RAMSES cell.
! Uniform run: dx == fine dx -> one cell (direct lookup). AMR: coarser cells
! average their fine cells. Periodic wrap on the fine indices.
!================================================================
subroutine gridic_condinit(x,q,dx,nn)
  use amr_parameters
  use hydro_parameters
  use gridic_data
  implicit none
  integer ::nn
  real(dp)::dx
  real(dp),dimension(1:nvector,1:nvar)::q
  real(dp),dimension(1:nvector,1:ndim)::x
  ! NB: local names must not collide with use-associated kind constants
  ! (amr_parameters has `sp`; Fortran is case-insensitive) -> sum_* prefixes.
  integer::i,ix,iy,iz,ix0,ix1,iy0,iy1,iz0,iz1,idx,ncov
  real(dp)::finedx,sum_r,sum_vx,sum_vy,sum_vz,sum_p,half

  finedx = gic_boxlen/dble(gic_nx)
  half = 0.5d0*dx
  do i=1,nn
     ix0=floor((x(i,1)-half)/finedx); ix1=ceiling((x(i,1)+half)/finedx)-1
     iy0=floor((x(i,2)-half)/finedx); iy1=ceiling((x(i,2)+half)/finedx)-1
#if NDIM>2
     iz0=floor((x(i,3)-half)/finedx); iz1=ceiling((x(i,3)+half)/finedx)-1
#else
     iz0=0; iz1=0
#endif
     sum_r=0d0; sum_vx=0d0; sum_vy=0d0; sum_vz=0d0; sum_p=0d0; ncov=0
     do ix=ix0,ix1
        do iy=iy0,iy1
           do iz=iz0,iz1
              idx=gic_idx(modulo(ix,gic_nx),modulo(iy,gic_ny),modulo(iz,gic_nz))
              sum_r=sum_r+gic_rho(idx); sum_vx=sum_vx+gic_vx(idx)
              sum_vy=sum_vy+gic_vy(idx); sum_vz=sum_vz+gic_vz(idx)
              sum_p=sum_p+gic_P(idx); ncov=ncov+1
           enddo
        enddo
     enddo
     q(i,1)=sum_r/dble(ncov)
     q(i,2)=sum_vx/dble(ncov)
#if NDIM>1
     q(i,3)=sum_vy/dble(ncov)
#endif
#if NDIM>2
     q(i,4)=sum_vz/dble(ncov)
#endif
     q(i,neul)=sum_p/dble(ncov)
  enddo
end subroutine gridic_condinit


!================================================================
! Stock analytic-disk-potential IC (kept so this file fully replaces the
! original condinit.f90 under PATCH).
!================================================================
subroutine ana_disk_potential_condinit(x,q,dx,nn)
  use amr_parameters
  use hydro_parameters
  use constants
  implicit none
  integer ::nn
  real(dp)::dx
  real(dp),dimension(1:nvector,1:nvar)::q
  real(dp),dimension(1:nvector,1:ndim)::x
  real(dp)::scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2
  integer::i
  real(dp)::height0=150
  real(dp)::dens0=0.66d0
  real(dp)::temp0=8000
  call units(scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)
  call region_condinit(x,q,dx,nn)
  do i=1,nn
     q(i,1) = dens0 * exp(-( x(i,ndim) - 0.5d0 * boxlen)**2 / (2.*height0**2))
     q(i,ndim+2) = q(i,1) * (kB*temp0/(mu_gas*mH))/scale_v**2
  end do
end subroutine ana_disk_potential_condinit
