import torch

ptq_path =  "ptq_weight.pth"
content_path = "ptq_weight.txt"

if __name__=="__main__":
    content_file = open(content_path,"wt")
    model = torch.load(ptq_path)
    for key in model.keys():
        item = model[key]
        print(key)
        for subkey in item.keys():
            print(subkey)
        # if hasattr(item,"dtype") and len(item.shape)>=1 and len(item)>1:
        #     print(key,item.shape,item.dtype,file=content_file)
        # else:
        #     print(key,item,file=content_file)