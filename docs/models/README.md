# 3D Printable Models (STL)

This directory contains the custom 3D-printable STL models for the **EezyBotArm** robotic cell: modular sorting table legs, NVIDIA Jetson Orin Nano protective enclosure, and the robot base M6 nut isolation cap.

---

## Model Index

| File | Description | Mounting / Application |
| :--- | :--- | :--- |
| [`table_leg_base.stl`](table_leg_base.stl) | Table leg base socket | Screws into the bottom wooden baseplate ($500 \times 300 \times 18\text{ mm}$) |
| [`table_leg_connector.stl`](table_leg_connector.stl) | Table leg connector insert | Screws into the underside of the elevated sorting table board ($300 \times 200 \times 9\text{ mm}$) |
| [`jetson_base.stl`](jetson_base.stl) | Jetson Orin Nano enclosure base | Houses the Jetson developer kit board with mounting standoffs for M3 heat-set inserts |
| [`jetson_lid.stl`](jetson_lid.stl) | Jetson Orin Nano enclosure lid | Top cover with ventilation, port cutouts, and 3x $12\text{ mm}$ front pushbutton mounts |
| [`m6_cap.stl`](m6_cap.stl) | Robot base M6 nut protective cap | Press-fits over the M6 mounting nut inside the arm base cavity to insulate PCA9685 |

---

## 1. Modular Table Legs

### Magnetic Quick-Release Mechanism
- Both the base socket (`table_leg_base.stl`) and the connector insert (`table_leg_connector.stl`) contain **two dedicated slots** for cylindrical neodymium magnets:
  - **Magnet dimensions**: $\varnothing 2\text{ mm} \times 1.5\text{ mm}$
- **Purpose**: Provides rigid, backlash-free seating during robot pick-and-place operations while allowing instant, tool-free table removal for maintenance, calibration, or transport.
- **Assembly tip**: Ensure magnetic polarities match (attractive force) before pressing or cyanoacrylate-gluing the magnets into the printed slots.

### Magnet-Free Friction-Fit Alternative
- Magnets are optional. If magnets are unavailable:
  - Print `table_leg_connector.stl` scaled at **~100.5%** in your slicer.
  - This slight enlargement provides a snug friction fit directly inside `table_leg_base.stl`.

### Fasteners & Board Thickness
- **Fasteners**: Mounted using $1.5 \times 6\text{ mm}$ self-tapping wood screws.
- **Board specs**:
  - Upper table board: $300 \times 200 \times 9\text{ mm}$
  - Bottom baseplate: $500 \times 300 \times 18\text{ mm}$
- > [!WARNING]
  > If using timber boards of different thicknesses, select screw lengths carefully to prevent screws from punching through the upper table surface.

### Workspace Parameter Coupling
- The leg height is precisely designed for a **9 mm table board** to establish a working plane at exactly **$90\text{ mm}$ ($0.09\text{ m}$)** above the baseplate (`table_z_m: 0.09` in [`environment.yaml`](../../src/eezybotarm_description/config/environment.yaml)).
- If different board thicknesses or leg geometries are used, update `table_z_m`, `table_width_m`, `table_depth_m`, or `robot_x_offset_m` in `environment.yaml` accordingly.

### Parametric CAD Source (Table Legs)
- 🔗 **[OnShape CAD Document: EezyBotArm Table Legs](https://cad.onshape.com/documents/16d39f2070b44c63ae6d65e0/w/bce775a624cd334bceeabdce/e/fb3a7e57508b0b67b010a788)**

---

## 2. Jetson Orin Nano Enclosure

A custom two-piece enclosure engineered specifically for the NVIDIA Jetson Orin Nano Developer Kit (and compatible carrier boards):

### Components
- **Base (`jetson_base.stl`)**: Features internal support ribs and alignment standoffs where the Jetson developer kit rests securely. Contains two dedicated pockets for brass heat-set threaded inserts.
- **Lid (`jetson_lid.stl`)**: Snaps onto the base with **2 snap-fit latches** at the rear/sides and fastens solidly at the front with **2 M3 machine screws**. Features:
  - Top ventilation cutouts for passive/active airflow.
  - Precise cutouts for all rear and side developer kit I/O ports (USB, Ethernet, DisplayPort, Barrel Jack, 40-Pin header routing).
  - Front panel mounting holes for three $12\text{ mm}$ momentary control buttons:
    - 🔴 **Power** (Red)
    - 🟢 **Force Recovery** (Green)
    - 🟡 **Reset** (Yellow)

### Hardware & Fasteners
- **Heat-Set Threaded Inserts**: 2x M3 brass inserts (length: $5\text{ mm}$, outer diameter: $\varnothing 4.2\text{ mm}$) heat-pressed into the front mounting posts of `jetson_base.stl` using a soldering iron.
- **Fastening Screws**: 2x M3 machine screws (e.g. M3 $\times 8\text{ mm}$ button head or countersunk) securing the lid to the base through the threaded inserts.
- **Mounting**: The enclosure simply rests on the wooden baseplate under the elevated sorting table between the legs, allowing easy removal without tools.

---

## 3. Arm Base M6 Nut Protective Cap (`m6_cap.stl`)

### Problem Solved
The EEZYbotARM Mk2 robotic arm chassis is bolted to the wooden baseplate using a central M6 threaded rod and M6 hex locknut located inside the bottom base cavity. 
In our build, the PCA9685 16-channel PWM driver board is also housed inside this arm base cavity directly above/adjacent to the central mounting bolt. Without protection, the exposed conductive M6 metal nut and bolt threads risk short-circuiting exposed solder pads on the underside of the PCA9685 board or abrading the PCB during arm rotation and vibration.

### Design & Installation
- **Dimensions**: Hexagonal inner cavity matching standard M6 nuts (across flats: $10\text{ mm}$) with an extended cylindrical dome (height: $14.5\text{ mm}$) to accommodate protruding bolt threads.
- **Installation**: Press-fit (friction-fit) directly onto the tightened M6 nut before placing and wiring the PCA9685 board. No adhesive or extra fasteners required.
- **Material**: Non-conductive PETG (or PLA). Provides mechanical standoff and reliable electrical insulation.
