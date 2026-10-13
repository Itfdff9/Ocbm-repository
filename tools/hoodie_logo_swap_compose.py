import json, subprocess, cv2, numpy as np, os
lm=json.load(open('landmarks_final.json'))
logo='logo_tight.png'; lw,lh=cv2.imread(logo,cv2.IMREAD_UNCHANGED).shape[1::-1]
place={}
for c,L in sorted(lm.items()):
    W=390 if c=='Royal-Seated' else 320
    cy=(L['tip']+L['pocket'])/2; cx=L['cx']
    tmp=f'final/_{c}.png'
    subprocess.run(['python3','/home/claude/royal/front_logo.py',f'clean/{c}.png',logo,tmp+'.jpg',str(cx),str(cy),str(W),'--quality','97'],check=True,capture_output=True)
    im=cv2.imread(tmp+'.jpg'); g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float)
    def content(r): return not (g[r].std()<8 and g[r].mean()>185)
    top=0
    while not content(top): top+=1
    bot=im.shape[0]-1
    while not content(bot): bot-=1
    top+=2; bot-=2   # skip antialiased edge rows
    out=im[top:bot+1]
    name=f"G18500_{'Royal_Seated' if c=='Royal-Seated' else c+'_Front'}_DREAMS.jpg"
    cv2.imwrite('final/'+name,out,[cv2.IMWRITE_JPEG_QUALITY,93]); os.remove(tmp+'.jpg')
    place[c]=dict(cx=cx,cy=round(cy,1)-top,width=W,file=name,size=out.shape[1::-1])
json.dump(place,open('final/placements.json','w'),indent=0)
print(len(place)); print({k:v['size'] for k,v in list(place.items())[:5]})
