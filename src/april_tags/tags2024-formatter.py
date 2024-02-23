import json
import re
import numpy as np

# LES DISTANCES SONT EN POUCE. LA CONVERSION EST FAITE PLUS BAS EN MULTIPLIANT PAR 2.54
data = "ID: 1, X: 593.68, Y: 9.68, Z: 53.38, ROTATION: 120°; ID: 2, X: 637.21, Y: 34.79, Z: 53.38, ROTATION:120°; ID: 3, X: 652.73, Y: 196.17, Z: 57.13, ROTATION:180°; ID: 4, X: 652.73, Y: 218.42, Z: 57.13, ROTATION:180°; ID: 5, X: 578.77, Y: 323.00, Z: 53.38, ROTATION:270°; ID: 6, X: 72.5, Y: 323.00, Z: 53.38, ROTATION:270°; ID: 7, X: -1.50, Y: 218.42, Z: 57.13, ROTATION:0°; ID: 8, X: -1.50, Y: 196.17, Z: 57.13, ROTATION:0°; ID: 9, X: 14.02, Y: 34.79, Z: 53.38, ROTATION:60°; ID: 10, X: 57.54, Y: 9.68 ,Z: 53.38, ROTATION:60°; ID: 11, X: 468.69, Y: 146.19, Z: 52.00, ROTATION:300°; ID: 12, X: 468.69, Y: 177.10, Z: 52.00, ROTATION:60°; ID: 13, X: 441.74, Y: 161.62, Z: 52.00, ROTATION:180°; ID: 14, X: 209.48, Y: 161.62, Z: 52.00, ROTATION:0°; ID: 15, X: 182.73, Y: 177.10, Z: 52.00, ROTATION:120°; ID: 16, X: 182.73, Y: 146.19, Z: 52.00, ROTATION:240°"

camera_oriented_matrix = [
                            [0,0,-1],
                            [1,0,0],
                            [0,-1,0]
                        ]

def formatField(f):
  [k, v] = f.split(':')
  v = re.sub("[^0-9.]", "", v.strip())
  v = float(v)

  return {'k': k.lower(), 'v': v}

formatted = []
for tag in data.split(';'):
  fields = []
  for field in tag.split(','):
    fields.append(formatField(field))

  id, x, y, z, ang_z = fields


  tag_data = {"pose": {"translation": {}, "rotation": {}}}
  tag_data[id['k'].upper().strip()] = int(id['v'])
  tag_data['pose']['translation'][x['k'].strip()] = x['v']*2.54
  tag_data['pose']['translation'][y['k'].strip()] = y['v']*2.54
  tag_data['pose']['translation'][z['k'].strip()] = z['v']*2.54

  rad_z = ang_z['v'] / 180 * np.pi

  # Rotation matrix around the z axis
  rot = [
    [np.cos(rad_z), -np.sin(rad_z), 0],
    [np.sin(rad_z), np.cos(rad_z), 0],
    [0, 0, 1]
  ]

  # multiply with the camera space conversion matrix
  rot_in_cam = np.around(np.matmul(rot, camera_oriented_matrix), 4)

  tag_data['pose']['rotation'] = rot_in_cam.tolist()

  formatted.append(tag_data)

file = json.dumps(formatted)
print(file)

