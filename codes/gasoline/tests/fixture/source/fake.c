#include <stdio.h>

#ifndef GASOLINE
#error "fixture requires the common GASOLINE define"
#endif

#define STRINGIFY_INNER(value) #value
#define STRINGIFY(value) STRINGIFY_INNER(value)

int main(void) {
#ifdef MHD
    printf("mhd=1 ");
#else
    printf("mhd=0 ");
#endif
#ifdef BAROTROPIC
    printf("barotropic=1 ");
#else
    printf("barotropic=0 ");
#endif
#ifdef EXTRA_VALUE
    printf("extra=%s ", STRINGIFY(EXTRA_VALUE));
#endif
    printf("\n");
    return 0;
}
