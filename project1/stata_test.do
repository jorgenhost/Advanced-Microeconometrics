cd "C:\Users\JBH\Dropbox\10_semester\Advanced Microeconometrics\project1"

use "firms.dta", clear

xtset firmid year

xtreg ldsa lcap lemp, fe
test lcap + lemp = 1

xtreg ldsa lcap lemp F.lemp, fe //strict exogeneity




reg ldsa lcap lemp
test lcap + lemp = 1
reg d.(ldsa lcap lemp), nocons //FD-est?
test d.lcap+d.lemp=1
 
 
 
//Test for auto correlation
xtserial ldsa lcap lemp //community written func
xtreg d.(ldsa lcap lemp)
predict eit, ue

gen eit_1 = l.eit
reg eit eit_1, nocons
test eit_1=-.5

binscatter eit eit_1

use "serial_corr.dta", clear
reg e e_1, nocons