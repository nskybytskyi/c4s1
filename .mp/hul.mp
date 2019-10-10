outputtemplate := "%j-%c.mps";
prologues := 3;

% cls && mpost hul.mp && epstopdf hul-1.mps && start hul-1.pdf
beginfig(1)

u := 2cm;
pair o, axis_s[], axis_xi[], t[];
o := (0, 0);
axis_s[0] := (-u/4, 0);
axis_s[1] := (u, 0);
axis_s[2] := (9u/8, 0);
axis_xi[0] := (0, -u/4);
axis_xi[1] := (0, u);
axis_xi[2] := (0, 9u/8);
t[1] := (3u/4, 0);
t[2] := (3u/4, 3u/4);
t[3] := (0, 3u/4);

fill o--t[2]--t[1]--cycle withcolor .9 white;
drawarrow (u/2,0)--(u/2,u/2);

% axis
label.llft(btex $0$ etex, o);
drawarrow axis_s[0]--axis_s[2];
label.bot(btex $s$ etex, axis_s[1]);
drawarrow axis_xi[0]--axis_xi[2];
label.lft(btex $\xi$ etex, axis_xi[1]);

% t
label.bot(btex $t$ etex, t[1]);
label.lft(btex $t$ etex, t[3]);
draw t[1]--t[2]--t[3] dashed evenly;
draw o--t[2];

endfig;

% cls && mpost hul.mp && epstopdf hul-2.mps && start hul-2.pdf
beginfig(2)

u := 3cm;
pair o, axis_x[];
o := (0, 0);
axis_x[0] := (-3u/4, 0);
axis_x[1] := (-u/2, 0);
axis_x[2] := (0, 0);
axis_x[3] := (u/2, 0);
axis_x[4] := (u, 0);
axis_x[5] := (5u/4, 0);
drawarrow axis_x[0]--axis_x[5];

% axis
label.llft(btex $-0.5$ etex, axis_x[1]);
label.bot(btex $0$ etex, o);
label.top(btex $0.5$ etex, axis_x[3]);
label.urt(btex $1$ etex, axis_x[4]);

% curve arrows
label.top(btex $I_0^{0.5}$ etex, (-u/4, u/8));
drawarrow o{dir 120} .. axis_x[1]{dir -120};
label.bot(btex $\frac{\mathrm{d}}{\mathrm{d} t}$ etex, (0, -17u/64));
drawarrow axis_x[1]{dir -60} .. axis_x[3]{dir 60};
label.top(btex $\frac{\mathrm{d}}{\mathrm{d} t}$ etex, (u/2, 17u/64));
drawarrow o{dir 60} .. axis_x[4]{dir -60};
label.bot(btex $I_0^{0.5}$ etex, (3u/4, -u/8));
drawarrow axis_x[4]{dir -120} .. axis_x[3]{dir 120};

endfig;
end
