/***********************************
* Disk of radius R_TR meshed in (x,y)
* Annulus R_TR < r < R meshed in (z,theta)
* with z = ln(r).
***********************************/
SetFactory("OpenCASCADE");

/** Input parameters: geometry **/

If (!Exists(R)) R=1; EndIf // radius circle
If (!Exists(R_TR)) R_TR=0.5*R; EndIf // truncation radius
xc = 0;
yc = 0;
x_offset = 0;
If (!Exists(y_offset)) y_offset = -R-Pi; EndIf
If (!Exists(lc)) lc=1/10; EndIf // char. length

/** Input parameters: mesh **/

N_TR = 5;
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;
//Mesh.ElementOrder = 2;

/** Geometry: circle **/

l_circle = newl; Circle(l_circle) = {xc, yc, 0, R_TR, 0, 2*Pi};
l_circleLoop = newl; Curve Loop(l_circleLoop) = {l_circle};
s_disk = news; Plane Surface(s_disk) = {l_circleLoop};

Physical Curve ("Disk-Boundary") = {l_circle};
Physical Surface ("Disk") = {s_disk};

/** Geometry: rectangle **/

p1 = newp; Point ( p1 ) = { Log(R_TR), y_offset+Pi, 0 };
p2 = newp; Point ( p2 ) = { Log(R), y_offset+Pi, 0 };
p3 = newp; Point ( p3 ) = { Log(R), y_offset-Pi, 0 };
p4 = newp; Point ( p4 ) = { Log(R_TR), y_offset-Pi, 0 };

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

Transfinite Curve {l_top, l_bot} = N_TR Using Progression 1;
Transfinite Surface {s_rect};

	// Periodicity condition Circle -> Rectangle
Periodic Curve {l_right} = {l_circle} ;
Periodic Curve {l_left} = {l_circle} ;

    //Periodicity polar coordinate (optional since transfinite)
//Periodic Curve {l_top} = {l_bot} ;
