import os

WORLD_FILE = "/home/satyajit/PPO_nav/src/rl_robot_gazebo/worlds/navigation.world"

with open(WORLD_FILE, "r") as f:
    content = f.read()

# Generate 60 walls
walls_xml = []
for i in range(60):
    xml = f"""
    <model name="wall_{i}">
      <pose>0 0 -10 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision"><geometry><box><size>1 1 1</size></box></geometry></collision>
        <visual name="visual"><geometry><box><size>1 1 1</size></box></geometry><material><ambient>0.8 0.1 0.1 1</ambient></material></visual>
      </link>
    </model>"""
    walls_xml.append(xml)

walls_str = "".join(walls_xml)
target = "<!-- We will use a script to generate the XML for 60 walls and overwrite this file. -->"
content = content.replace(target, walls_str)

with open(WORLD_FILE, "w") as f:
    f.write(content)

print("Generated 60 walls in navigation.world")
