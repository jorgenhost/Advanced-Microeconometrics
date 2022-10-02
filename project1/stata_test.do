cd "C:\Users\JBH\Dropbox\10_semester\Advanced Microeconometrics\repo\project1"

use "firms.dta", clear

xtset firmid

xtreg ldsa lcap lemp, fe
xtreg ldsa lcap lemp i.year, fe


outreg

binscatter(ldsa lemp)
binscatter(ldsa lcap)