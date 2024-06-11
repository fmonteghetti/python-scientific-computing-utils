/***********************************
* Rectangle width a height b
***********************************/
SetFactory("OpenCASCADE");

/** Input parameters: geometry **/

x_offset = 0;
y_offset = 0;
If (!Exists(x_l)) x_l=0; EndIf 
If (!Exists(x_r)) x_r=2; EndIf 
If (!Exists(y_t)) y_t=1; EndIf 
If (!Exists(y_b)) y_b=0; EndIf 

/** Input parameters: mesh **/
If (!Exists(N_x)) N_x=100; EndIf 
If (!Exists(N_y)) N_y=100; EndIf 
Mesh.CharacteristicLengthMin = 0.01;
Mesh.CharacteristicLengthMax = 0.01;
//Mesh.ElementOrder = 2;

/** Geometry: rectangle **/

p1 = newp; Point ( p1 ) = { x_l, y_t, 0 };
p2 = newp; Point ( p2 ) = { x_r, y_t, 0 };
p3 = newp; Point ( p3 ) = { x_r, y_b, 0 };
p4 = newp; Point ( p4 ) = { x_l, y_b, 0 };

l_top = newl; Line(l_top) = {p1, p2};
l_right = newl; Line(l_right) = {p2, p3};
l_bot = newl; Line(l_bot) = {p3, p4};
l_left = newl; Line(l_left) = {p4, p1};

l_rectLoop = newl; Curve Loop ( l_rectLoop ) = { l_top,l_right,l_bot,l_left };
s_rect = news; Plane Surface ( s_rect ) = { l_rectLoop };

Physical Curve ("Rectangle-Boundary-Top") = {l_top};
Physical Curve ("Rectangle-Boundary-Right") = {l_right};
Physical Curve ("Rectangle-Boundary-Left") = {l_left};
Physical Curve ("Rectangle-Boundary-Bot") = {l_bot};

Physical Surface ("Rectangle") = {s_rect};

/** Mesh: rectangle **/

Transfinite Curve {l_top, l_bot} = N_x Using Progression 1;
Transfinite Curve {l_left, l_right} = N_y Using Progression 1;
Transfinite Surface {s_rect};
