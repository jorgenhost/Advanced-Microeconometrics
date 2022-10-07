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
 
outreg

binscatter(ldsa lemp)
binscatter(ldsa lcap)

use "serial_corr.dta", clear
reg e e_1