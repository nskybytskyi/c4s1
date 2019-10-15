outputtemplate := "%j-%c.mps";
prologues := 3;

% cls && mpost rai.mp && epstopdf rai-1.mps && start rai-1.pdf
beginfig(1)

u := 2cm;
pair o, axis_x[], axis_y[], f[];
o := (0, 0);
axis_x[0] := (-u/4, 0);
axis_x[1] := (u/2, 0);
axis_x[2] := (u, 0);
axis_x[3] := (3u/2, 0);
axis_x[4] := (7u/4, 0);
axis_y[0] := (0, -u/4);
axis_y[1] := (0, u);
axis_y[2] := (0, 5u/4);
f[0] := o;
f[1] := axis_x[1];
f[2] := (3u/4, u);
f[3] := axis_x[2];
f[4] := axis_x[3];

% axis
label.llft(btex $0$ etex, o);
drawarrow axis_x[0]--axis_x[4];
label.bot(btex $1$ etex, axis_x[3]);
drawarrow axis_y[0]--axis_y[2];
label.lft(btex $1$ etex, axis_y[1]);

% f
label.bot(btex $x_{i-1}$ etex, f[1]);
label.bot(btex $x_i$ etex, f[3]);
draw axis_y[1]--f[2] dashed evenly;
draw f[0]--f[1]--f[2]--f[3]--f[4] withcolor red;

draw f[0] withcolor black withpen pencircle scaled 2bp;
draw f[4] withcolor black withpen pencircle scaled 2bp;

endfig;
end
