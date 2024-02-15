let data = {
  "tags": [
    {
      "ID": 1,
      "pose": {
        "translation": {
          "x": 593.68,
          "y": 9.68,
          "z": 53.38
        },
        "rotation": [
          [-0.5, -0.87, 0],
          [0.87, -0.5, 0],
          [0, 0, 1]
        ]
      }
    },
    {
      "ID": 2,
      "pose": {
        "translation": {
          "x": 637.21,
          "y": 34.79,
          "z": 53.38
        },
        "rotation": [
          [-0.5, -0.87, 0],
          [0.87, -0.5, 0],
          [0, 0, 1]
        ]
      }
    },
    {
      "ID": 3,
      "pose": {
        "translation": {
          "x": 652.73,
          "y": 196.17,
          "z": 57.13
        },
        "rotation": [
          [-1, 0, 0],
          [0, -1, 0],
          [0, 0, 1]
        ]
      }
    },
    {
      "ID": 4,
      "pose": {
        "translation": {
          "x": 652.73,
          "y": 218.42,
          "z": 57.13
        },
        "rotation": [
          [-1, 0, 0],
          [0, -1, 0],
          [0, 0, 1]
        ]
      }
    },
    {
      "ID": 5,
      "pose": {
        "translation": {
          "x": 578.77,
          "y": 323.00,
          "z": 53.38
        },
        "rotation": [
          [0, 1, 0],
          [-1, 0, 0],
          [0, 0, 1]
        ]
      }
    },
    {
      "ID": 6,
      "pose": {
        "translation": {
          "x": 72.5,
          "y": 323.00,
          "z": 53.38
        },
        "rotation": [
          [0, 1, 0],
          [-1, 0, 0],
          [0, 0, 1]
        ]
      }
    },
    {
      "ID": 7,
      "pose": {
        "translation": {
          "x": -1.50,
          "y": 218.42,
          "z": 57.13
        },
        "rotation": [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]
        ]
      }
    },
    {
      "ID": 8,
      "pose": {
        "translation": {
          "x": -1.50,
          "y": 196.17,
          "z": 57.13
        },
        "rotation": [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]
        ]
      }
    }
  ],
  "field": {
    "length": 16.54175,
    "width": 8.21055
  }
}

// Convert existing data to cm
for(tag of data.tags){
  coords = tag.pose.translation
  for(k in coords){
    coords[k] = (coords[k]*0.0254).toFixed(4); // convert inches to m
  }
}

// Format rest of data
function formatField(f){
  [k, v] = f.split(':')
  v = parseFloat(v)

  return {'k': k.toLowerCase(), 'v': v}
}

rest_of_data = "ID: 9, X: 14.02, Y: 34.79, Z: 53.38, ROTATION:60°; ID: 10, X: 57.54, Y: 9.68 ,Z: 53.38, ROTATION:60°; ID: 11, X: 468.69, Y: 146.19, Z: 52.00, ROTATION:300°; ID: 12, X: 468.69, Y: 177.10, Z: 52.00, ROTATION:60°; ID: 13, X: 441.74, Y: 161.62, Z: 52.00, ROTATION:180°; ID: 14, X: 209.48, Y: 161.62, Z: 52.00, ROTATION:0°; ID: 15, X: 182.73, Y: 177.10, Z: 52.00, ROTATION:120°; ID: 16, X: 182.73, Y: 146.19, Z: 52.00, ROTATION:240°"
formatted = []
for(tag of rest_of_data.split(';')){
  fields = []
  for(field of tag.split(',')){
    fields.push(formatField(field))
  }

  [id, x, y, z, ang_z] = fields;

  tag_data = {"pose": {"translation": {}, "rotation": {}}}
  tag_data[id.k.toUpperCase()] = id.v
  tag_data.pose.translation[x.k] = x.v
  tag_data.pose.translation[y.k] = y.v
  tag_data.pose.translation[z.k] = z.v

  rad_z = ang_z.v / 180 * Math.PI

  // Rotation matrix around the z axis
  tag_data.pose.rotation = [
    [Number(Math.cos(rad_z).toFixed(2)), -Number(Math.sin(rad_z).toFixed(2)), 0],
    [Number(Math.sin(rad_z).toFixed(2)), Number(Math.cos(rad_z).toFixed(2)), 0],
    [0, 0, 1]
  ]

  formatted.push(tag_data)
}

data.tags.push(formatted)

console.log(JSON.stringify(data))