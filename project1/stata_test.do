cd "C:\Users\JBH\Dropbox\10_semester\Advanced Microeconometrics\repo\project1"

use "firms.dta", clear

xtset firmid

xtreg ldsa lcap lemp i.year, fe

xtreg ldsa lcap c.lemp##i.year, fe