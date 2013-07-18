#include "mmm1dgpu_tcl.hpp"
#include "forces.hpp"
#include "interaction_data.hpp"
#include "Mmm1dgpuForce.hpp"

#ifdef MMM1D_GPU

int tclcommand_inter_coulomb_parse_mmm1dgpu(Tcl_Interp *interp, int argc, char **argv)
{
  double switch_rad, maxPWerror;
  int bessel_cutoff;

  if (argc < 2) {
    Tcl_AppendResult(interp, "wrong # arguments: inter coulomb mmm1dgpu <switch radius> "
		     "{<bessel cutoff>} <maximal error for near formula> | tune  <maximal pairwise error>", (char *) NULL);
    return TCL_ERROR;
  }

  if (ARG0_IS_S("tune")) {
    /* autodetermine bessel cutoff AND switching radius */
    if (! ARG_IS_D(1, maxPWerror))
      return TCL_ERROR;
    bessel_cutoff = -1;
    switch_rad = -1;
  }
  else {
    if (argc == 2) {
      /* autodetermine bessel cutoff */
      if ((! ARG_IS_D(0, switch_rad)) ||
	  (! ARG_IS_D(1, maxPWerror))) 
	return TCL_ERROR;
      bessel_cutoff = -1;
    }
    else if (argc == 3) {
      /* fully manual */
      if((! ARG_IS_D(0, switch_rad)) ||
	 (! ARG_IS_I(1, bessel_cutoff)) ||
	 (! ARG_IS_D(2, maxPWerror))) 
	return TCL_ERROR;

      if (bessel_cutoff <=0) {
	Tcl_AppendResult(interp, "bessel cutoff too small", (char *)NULL);
	return TCL_ERROR;
      }
    }
    else {
      Tcl_AppendResult(interp, "wrong # arguments: inter coulomb mmm1dgpu <switch radius> "
		       "{<bessel cutoff>} <maximal error for near formula> | tune  <maximal pairwise error>", (char *) NULL);
      return TCL_ERROR;
    }
    
    if (switch_rad <= 0 || switch_rad > box_l[2]) {
      Tcl_AppendResult(interp, "switching radius is not between 0 and box_l[2]", (char *)NULL);
      return TCL_ERROR;
    }
  }
  
  // TODO: when does coulomb.prefactor change?
  Mmm1dgpuForce *A = new Mmm1dgpuForce(coulomb.prefactor, maxPWerror, switch_rad, bessel_cutoff);
  // using new makes sure it doesn't get destroyed when we leave tclcommand_inter_coulomb_parse_mmm1dgpu
  FI.addMethod(A);

  return TCL_OK;
}

#endif