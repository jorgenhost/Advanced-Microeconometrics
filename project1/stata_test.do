cd "C:\Users\JBH\Dropbox\10_semester\Advanced Microeconometrics\repo\project1"

use "firms.dta", clear

xtset firmid year

xtreg ldsa lcap lemp, fe
xtreg ldsa lcap lemp i.year, fe
reg d.(ldsa lcap lemp), nocons //FD-est?
 
outreg

binscatter(ldsa lemp)
binscatter(ldsa lcap)