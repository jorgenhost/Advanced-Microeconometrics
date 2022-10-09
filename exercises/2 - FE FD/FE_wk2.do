cd "C:\Users\JBH\Dropbox\10_semester\Advanced Microeconometrics\exercises\2 - FE FD"

use "FE_wk2.dta", clear

xtset ID year

xtreg log_wage Experience Experience_sqr Married Union, fe
predict eit, ue

gen eit_1 = l.eit

reg eit eit_1, nocons

corr eit eit_1