from PIL import Image,ImageDraw;import glob,os
S='data/tmp/labeller_scratch/'
for f in sorted(glob.glob('tests/fixtures/crop/real/*.jpg')):
    n=os.path.basename(f)
    if n.startswith('orient'):continue
    im=Image.open(f).convert('RGB');d=ImageDraw.Draw(im);w,h=im.size
    for x in range(0,w,100):
        d.line([x,0,x,h],fill=(255,255,0) if x%500==0 else (0,255,255),width=1);d.text((x+2,2),str(x),fill=(255,255,0))
    for y in range(0,h,100):
        d.line([0,y,w,y],fill=(255,255,0) if y%500==0 else (0,255,255),width=1);d.text((2,y+2),str(y),fill=(255,255,0))
    im.save(S+'g_'+n)
