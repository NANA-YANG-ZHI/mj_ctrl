// Solid arch — radius = 100 mm, length = 100 mm along Y

radius = 100;  // mm (0.1 m)
length = 100;  // mm along Y

intersection() {
    rotate([90, 0, 0])
        cylinder(h=length, r=radius, center=true, $fn=128);
    translate([0, 0, radius/2])
        cube([2*radius, length, radius], center=true);
}
