cls
cd "C:\Users\JBH\Dropbox\10_semester\Advanced Microeconometrics\project1"

use "firms.dta", clear

xtset firmid year

//POLS
xtreg ldsa lemp lcap, noconstant cluster(firmid)
xtreg ldsa lemp lcap, robust

// Fixed effect
xtreg ldsa lemp lcap, fe
test lcap + lemp = 1

// First difference
xtreg d.(ldsa lemp lcap)
test d.lcap+d.lemp=1

// First difference (robust)
xtreg d.(ldsa lemp lcap), robust
test d.lcap+d.lemp=1
 
// Fixed effect strict exo test
xtreg ldsa lemp lcap F.lemp F.lcap, fe
test F.lemp F.lcap

// First difference strict exo test
xtreg d.(ldsa lcap lemp) lcap lemp
test lcap lemp


//Test for auto correlation
quietly: reg d.(ldsa lcap lemp)
predict eit, res

gen eit_1 = l.eit

reg eit eit_1, nocons
test eit_1=-.5 	// u_its are serially uncorrelated 	(pref FE)
test eit_1=0	// u_its are a random walk			(pref FD)