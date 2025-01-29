set terminal png size 480,360

set output 'shift_comp.png'
set title 'comparison of Shift implementations'
set key left top
unset colorbox
set palette defined (0 "white", 1 "dark-red")
set xlabel 'n'
set ylabel 'depth'
#set xtics (2,3,4,5,6)
plot "MCX_Shift_gate.dat" u 2:8 w l title "MCX" lt palette frac 0.15,\
     "QFT_Shift_gate.dat" u 2:8 w l title "QFT" lt palette frac 0.5,\
     "QFT_Shift_gate_MCX_Shift_gate.dat" u 2:8 w l title "Min" lt palette frac 1
