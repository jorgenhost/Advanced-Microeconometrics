cls
cd "C:\Users\JBH\Dropbox\10_semester\Advanced Microeconometrics\project1"

use "firms.dta", clear

xtset firmid year

// Fixed effect
xtreg ldsa lemp lcap, fe
test lcap + lemp = 1

// First difference
reg d.(ldsa lemp lcap), nocons
test d.lcap+d.lemp=1
 
// Fixed effect strict exo test
xtreg ldsa lemp lcap F.lemp F.lcap, fe
test F.lemp F.lcap

// First difference strict exo test
reg d.(ldsa lcap lemp) lcap lemp
test lcap lemp


//Test for auto correlation
quietly: reg d.(ldsa lcap lemp)
predict eit, res

gen eit_1 = l.eit

reg eit eit_1, nocons
test eit_1=-.5 	// u_its are serially uncorrelated
test eit_1=0	// u_its are a random walk

use "serial_corr.dta", clear
reg e e_1, nocons