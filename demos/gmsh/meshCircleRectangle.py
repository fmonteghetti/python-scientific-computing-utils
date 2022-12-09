########################################################
## Mesh circle (0,R)
# Mesh 1: unstructured (x,y)
# Mesh 2: unstructured (x,y) + Euler (z,theta) 
########################################################

import gmsh
import sys
import math as mt
pi = mt.pi

gmsh.initialize()
gmsh.logger.start()
gmsh.option.setNumber("General.Terminal", 1)

gmsh.model.add("circleRectangle")
R = 1;
lc = 1e-1;
R_TR = R/2;
y_offset = -2*R-pi;

# Circle (0,R_TR)
l_circle = gmsh.model.occ.addCircle(0, 0, 0, R_TR);
l_circleLoop = gmsh.model.occ.addCurveLoop([l_circle]);
s_disk = gmsh.model.occ.addPlaneSurface([l_circleLoop]);

# Rectangle (ln(R_TR),y_offset+pi) -------- (ln(R), y_offset+pi)
#           (ln(R_TR),y_offset-pi) -------- (ln(R), y_offset-pi)      
#l_rect = gmsh.model.occ.addRectangle(mt.log(R_TR), y_offset+pi, 0, mt.log(R/R_TR), -2*pi);
#s_rect = gmsh.model.occ.addPlaneSurface([l_rect]);

p1  = gmsh.model.occ.addPoint(mt.log(R_TR), y_offset+pi, 0)
p2  = gmsh.model.occ.addPoint(mt.log(R), y_offset+pi, 0)
p3  = gmsh.model.occ.addPoint(mt.log(R), y_offset-pi, 0)
p4  = gmsh.model.occ.addPoint(mt.log(R_TR), y_offset-pi, 0)

l_top = gmsh.model.occ.addLine(p1, p2)
l_right = gmsh.model.occ.addLine(p2, p3)
l_bot = gmsh.model.occ.addLine(p3, p4)
l_left = gmsh.model.occ.addLine(p4, p1)
l_rectLoop = gmsh.model.occ.addCurveLoop([l_top,l_right,l_bot,l_left])
s_rect = gmsh.model.occ.addPlaneSurface([l_rectLoop])

gmsh.model.occ.synchronize()


tmp = gmsh.model.addPhysicalGroup(1, [l_top])
gmsh.model.setPhysicalName(1, tmp, "Rectangle-Top")
tmp = gmsh.model.addPhysicalGroup(1, [l_bot])
gmsh.model.setPhysicalName(1, tmp, "Rectangle-Bottom")
pg_left = gmsh.model.addPhysicalGroup(1, [l_left])
gmsh.model.setPhysicalName(1, pg_left, "Rectangle-Left")
tmp = gmsh.model.addPhysicalGroup(1, [l_right])
gmsh.model.setPhysicalName(1, tmp, "Rectangle-Right")
pg_circle = gmsh.model.addPhysicalGroup(1, [l_circleLoop])
gmsh.model.setPhysicalName(1, pg_circle, "Circle-Boundary")
tmp = gmsh.model.addPhysicalGroup(2, [s_disk])
gmsh.model.setPhysicalName(2, tmp, "Disk")
tmp = gmsh.model.addPhysicalGroup(2, [s_rect])
gmsh.model.setPhysicalName(2, tmp, "Rectangle")


gmsh.model.occ.synchronize()

#gmsh.write("model.geo")
gmsh.model.mesh.setTransfiniteCurve(l_top, 5)
gmsh.model.mesh.setTransfiniteCurve(l_bot, 5)
#gmsh.model.mesh.setTransfiniteCurve(l_left, 5)
#gmsh.model.mesh.setTransfiniteCurve(l_right, 5)
    # How to reproduce the call 'Periodic Curve {l_left} = {l_circleLoop} ;' ??
    # Option 'Periodic Curve' defined in Gmsh.y
    # Python setPeriodic
    # calls gmshModelMeshSetPeriodic (gmshc.cpp)
    # calls gmsh::model::mesh::setPeriodic(dim,tag,tagM,affineTransfo) (gmsh.cpp)
    # check that size is 16
    #calls target->setMeshMaster(source,affineTransfor) (GEdge.cpp)
gmsh.model.mesh.setPeriodic(1,[l_left],[l_circle],[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
#gmsh.model.mesh.setPeriodic(1,[l_right],[l_left],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

#gmsh.model.mesh.setPeriodic(1,[l_right],[l_circleLoop],[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

#gmsh.model.mesh.setTransfiniteSurface(s_rect)

gmsh.model.occ.synchronize()

nodeP = gmsh.model.mesh.getPeriodicNodes(1, l_left)

#gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)
#gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)
#gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)


# Check order in GUI using Mesh\Inspect on boundary triangles
gmsh.model.mesh.setOrder(2)
gmsh.option.setNumber("Mesh.ElementOrder", 2)
gmsh.model.mesh.generate(2)

# ... and save it to disk
#gmsh.write("t2.msh")

# TODO: check periodicity quantitively
nodeLeft = gmsh.model.mesh.getNodesForPhysicalGroup(1, pg_left)


# Inspect the log:
log = gmsh.logger.get()
gmsh.logger.stop()

print("Logger has recorded " + str(len(log)) + " lines")
print(*log, sep = "\n") 

gmsh.fltk.run()




gmsh.finalize()
