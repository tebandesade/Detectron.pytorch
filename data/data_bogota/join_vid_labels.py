import argparse
import os 
import sys
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a Bogota labels into one label file')
    parser.add_argument(
        '--dir_json_files', dest='directory_json_file',
        help='Pretrained network weights file path',
        default=None, type=str)
    parser.add_argument(
        '--output', dest='out_file_name',
        help='Output file path',
        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def parse_vgg_bogota(file_):
    with open(file_) as fil:
        datos = json.load(fil)

        data_images = datos['_via_img_metadata']
        return data_images

        #print(type(data_images))
        #if list then just get that value for each json
        #append values 
        # new dictionary item with _via_img with values appended
        '''
        for k, v in data_images.items():
            name_ = v['filename']
            for region in v['regions']:
                reg_shp = region['shape_attributes']
                label_ = region['region_attributes']['type']
                x  = reg_shp['x']
                y  = reg_shp['y']
                h  = reg_shp['width']
                w  = reg_shp['height']
                bounding = [x,y,h,w]
                ret = {}
                temp = {'bbox':bounding,'label':label_}
                ret[name_] = temp 
                lista_final.append(ret)
         '''
def load_and_convert(args):
	header_ruta = args.directory_json_file
	archivos = os.listdir(header_ruta)
	dic_final = {}
	for arch in archivos:
		rta_file = os.path.join(header_ruta,arch)
		data = parse_vgg_bogota(rta_file)
		dic_final.update(data)

	return dic_final

if __name__ == '__main__':
    args = parse_args()
    print(args)
    assert os.path.exists(args.directory_json_file), \
        'Directory does not exist'
    dic_final = load_and_convert(args)
    dic= {}
    dic['_via_img_metadata'] = dic_final
    out = args.out_file_name + '.json'
    with open(out, "w") as fp:
    	json.dump(dic , fp) 
