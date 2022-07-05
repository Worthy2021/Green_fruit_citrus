import json 

path = './test'

import os 
files = []
for file in os.listdir(path):
    if file[-5:] == '.json':
        files.append(file)

print(json.load(open('./test/'+files[0])))

via_region_data = {}

for file in files:
    one_json = json.load(open('./test/'+file))
    
    one_image = {}
    one_image['filename'] = file.split('.')[0]+'.jpg'
    
    shape = one_json['shapes']
    
    regions = {}
    
    for i in range(len(shape)):
        
        import numpy as np
        points = np.array(shape[i]['points'])
        
        all_points_x = list(points[:,0])
        all_points_y = list(points[:,1])
        
        regions[str(i)] = {}
        regions[str(i)]['region_attributes'] = {}
        regions[str(i)]['shape_attributes'] = {}
        
        regions[str(i)]['shape_attributes']['all_points_x'] = all_points_x
        regions[str(i)]['shape_attributes']['all_points_y'] = all_points_y
        regions[str(i)]['shape_attributes']['name'] = shape[i]['label']
        
    one_image['regions'] = regions
    one_image['size'] = 0
    
    via_region_data[file] = one_image
    

with open('./images/via_region_data.json','w') as f:
    json.dump(via_region_data,f,sort_keys=False,ensure_ascii=True,indent=(2))
       
        