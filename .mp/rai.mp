outputtemplate := "%j-%c.mps";
prologues := 3;

% cls && mpost rai.mp && epstopdf rai-3.mps && start rai-3.pdf
beginfig(3)

u := 2cm;
pair o, axis_x[], axis_y[], f[];
o := (0, 0);
axis_x[0] := (-u/4, 0);
axis_x[1] := (u/2, 0);
axis_x[2] := (u, 0);
axis_x[3] := (3u/2, 0);
axis_x[4] := (7u/4, 0);
axis_y[0] := (0, -u/4);
axis_y[1] := (0, u/2);
axis_y[2] := (0, u);
axis_y[3] := (0, 5u/4);
f[0] := axis_x[1] + axis_y[1];
f[1] := axis_x[2] + axis_y[1];
f[2] := axis_x[3] + axis_y[1];
f[3] := axis_x[2] + axis_y[2];

% axis
drawarrow axis_x[0]--axis_x[4]+(u/16,0);
label.bot(btex $x_{i - 1}$ etex, axis_x[1]);
label.bot(btex $x_i$ etex, axis_x[2]);
label.bot(btex $x_{i + 1}$ etex, axis_x[3]);
drawarrow axis_y[0]--axis_y[3];
label.lft(btex $t_j$ etex, axis_y[1]);
label.lft(btex $t_{j + 1}$ etex, axis_y[2]);

% f
draw axis_x[1]--axis_x[1]+axis_y[3] dashed evenly;
draw axis_x[2]--axis_x[2]+axis_y[3] dashed evenly;
draw axis_x[3]--axis_x[3]+axis_y[3] dashed evenly;
draw axis_y[1]--axis_y[1]+axis_x[4] dashed evenly;
draw axis_y[2]--axis_y[2]+axis_x[4] dashed evenly;
fill fullcircle scaled 2.5 shifted f[0] withcolor black;
fill fullcircle scaled 2.5 shifted f[1] withcolor black;
draw fullcircle scaled 4.5 shifted f[1] withpen pencircle scaled 1bp;
fill fullcircle scaled 2.5 shifted f[2] withcolor black;
fill fullcircle scaled 2.5 shifted f[3] withcolor black;
draw fullcircle scaled 4.5 shifted f[3] withpen pencircle scaled 1bp;
endfig;

% cls && mpost rai.mp && epstopdf rai-4.mps && start rai-4.pdf
beginfig(4)

u := 2cm;
pair o, axis_x[], axis_y[], f[];
o := (0, 0);
axis_x[0] := (-u/4, 0);
axis_x[1] := (u/2, 0);
axis_x[2] := (u, 0);
axis_x[3] := (3u/2, 0);
axis_x[4] := (7u/4, 0);
axis_y[0] := (0, -u/4);
axis_y[1] := (0, u/2);
axis_y[2] := (0, u);
axis_y[3] := (0, 5u/4);
f[0] := axis_x[1] + axis_y[2];
f[1] := axis_x[2] + axis_y[1];
f[2] := axis_x[3] + axis_y[2];
f[3] := axis_x[2] + axis_y[2];

% axis
drawarrow axis_x[0]--axis_x[4]+(u/16,0);
label.bot(btex $x_{i - 1}$ etex, axis_x[1]);
label.bot(btex $x_i$ etex, axis_x[2]);
label.bot(btex $x_{i + 1}$ etex, axis_x[3]);
drawarrow axis_y[0]--axis_y[3];
label.lft(btex $t_j$ etex, axis_y[1]);
label.lft(btex $t_{j + 1}$ etex, axis_y[2]);

% f
draw axis_x[1]--axis_x[1]+axis_y[3] dashed evenly;
draw axis_x[2]--axis_x[2]+axis_y[3] dashed evenly;
draw axis_x[3]--axis_x[3]+axis_y[3] dashed evenly;
draw axis_y[1]--axis_y[1]+axis_x[4] dashed evenly;
draw axis_y[2]--axis_y[2]+axis_x[4] dashed evenly;
fill fullcircle scaled 2.5 shifted f[0] withcolor black;
fill fullcircle scaled 2.5 shifted f[1] withcolor black;
draw fullcircle scaled 4.5 shifted f[1] withpen pencircle scaled 1bp;
fill fullcircle scaled 2.5 shifted f[2] withcolor black;
fill fullcircle scaled 2.5 shifted f[3] withcolor black;
draw fullcircle scaled 4.5 shifted f[3] withpen pencircle scaled 1bp;
endfig;
end
