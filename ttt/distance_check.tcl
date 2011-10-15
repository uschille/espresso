setmd skin 0.1
setmd box_l 40 40 40
setmd time_step 0.1
cellsystem domain_decomposition -no_verlet_list

lbfluid den 1 agrid 1. tau 0.1 visc 1.0 ext_force 0 0 .002

thermostat lb 0.
lbboundary rhomboid corner 15 15 15 a 10 0 0 b 0 10 0 c 0 0 10 direction outside
lbfluid print vtk boundary boundary.vtk
exit

for {set i 0} {$i < 500} {incr i 10} {
	if {$i % 10 == 0} {
		puts -nonewline "integrating $i"
		puts "th step"
		flush stdout
	}
	
  integrate 10
	lbfluid print vtk velocity velocity_$i.vtk	
}
