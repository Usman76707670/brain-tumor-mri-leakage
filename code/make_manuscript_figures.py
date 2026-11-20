# Redraws the manuscript figures that are built from released files (Figures 1(a), 2 to 5 and 9).
# Run from the repository root: python code/make_manuscript_figures.py ; Figure 1(a) is drawn with: dot -Tpng -Gdpi=170 results/figures/wf.dot -o results/figures/fig1a_workflow.png
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
R='.'; O='results/figures/'
m=json.load(open(R+'/logs/runs_patient/meta_seed42.json'))
for key,name,fn in [('swin','Swin-V2-B','fig2_swin_curves.png'),('resnet','ResNet101','fig3_resnet_curves.png')]:
    h=m[key]['hist']; be=m[key]['best_epoch']; ep=np.arange(1,13)
    fig,ax=plt.subplots(1,2,figsize=(12,4.3))
    ax[0].plot(ep,h['train_loss'],'o-',label='Training loss'); ax[0].plot(ep,h['val_loss'],'s-',label='Validation loss')
    ax[1].plot(ep,np.array(h['train_acc'])*100,'o-',label='Training accuracy'); ax[1].plot(ep,np.array(h['val_acc'])*100,'s-',label='Validation accuracy')
    for a,t,yl in [(ax[0],'Loss','Cross-entropy loss (label smoothing 0.1)'),(ax[1],'Accuracy','Accuracy (%)')]:
        a.axvline(be,ls=':',c='gray',label=f'Epoch kept ({be})'); a.set_title(f'{name}: {t} (seed 42, patient-aware split)')
        a.set_xlabel('Epoch'); a.set_ylabel(yl); a.set_xticks(ep); a.grid(alpha=.3); a.legend()
    plt.tight_layout(); plt.savefig(O+fn,dpi=200); plt.close()
d=pd.read_csv(R+'/predictions/test_predictions_all_seeds.csv.gz'); P=['p_Glioma','p_Meningioma','p_No Tumor','p_Pituitary']
C=['Glioma','Meningioma','No Tumor','Pituitary']
def get(mm,s,v='orig'):
    x=d[(d.model==mm)&(d.seed==s)&(d.view==v)].sort_values('file'); return x.label.values,x[P].values
for mm,name,fn in [('resnet','ResNet101 branch, no TTA','fig4_cm_resnet.png'),('swin','Swin-V2-B branch, no TTA','fig5_cm_swin.png')]:
    y,p=get(mm,42); pr=p.argmax(1); cm=confusion_matrix(y,pr)
    plt.figure(figsize=(6.8,5.6)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=C,yticklabels=C,annot_kws={'size':14})
    plt.title(f'{name} (seed 42): {(pr==y).sum()}/{len(y)} = {(pr==y).mean()*100:.2f}%'); plt.xlabel('Predicted label'); plt.ylabel('True label')
    plt.tight_layout(); plt.savefig(O+fn,dpi=200); plt.close()
# ablation
ab=pd.read_csv(R+'/results/internal/patient_table_ablation_baselines.csv').set_index('Configuration')
groups=[('ResNet101','ResNet101'),('Swin-V2-B','Swin-V2-B'),('Equal fusion','Equal\nfusion'),('Fixed 46/54 fusion','Fixed 46/54\nfusion'),('Tuned fusion','Tuned\nfusion'),('EfficientNet-B0','EfficientNet-B0'),('ConvNeXt-T','ConvNeXt-T'),('ResNet50','ResNet50')]
fig,ax=plt.subplots(figsize=(13,5.2)); x=np.arange(len(groups)); wd=.38
for j,(suf,col,lab) in enumerate([(', no TTA','#9db7e0','Without TTA (primary)'),(', TTA','#2f5597','With TTA')]):
    mu=[];sd=[];xs=[]
    for i,(g,_) in enumerate(groups):
        k=g+suf
        if k in ab.index: mu.append(ab.loc[k,'Accuracy % mean']); sd.append(ab.loc[k,'SD']); xs.append(i+(j-.5)*wd)
    ax.bar(xs,mu,wd,yerr=sd,capsize=4,color=col,label=lab)
    for xx,mm_,ss in zip(xs,mu,sd): ax.text(xx,mm_+ss+.12,f'{mm_:.2f}',ha='center',fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([g[1] for g in groups]); ax.set_ylim(86,96); ax.set_ylabel('Test accuracy (%), mean ± SD over 4 seeds')
ax.set_title('Ablation and baselines on the patient-aware split (623 test images)'); ax.grid(axis='y',alpha=.3); ax.legend(loc='upper right')
plt.tight_layout(); plt.savefig(O+'fig9_ablation.png',dpi=200); plt.close()
# Grad-CAM random grid rearranged 4 x 6 (needs the notebook output Fig_GradCAM_random_seed42.png, which contains MRI images and is not in this repository)
import os
SRC='Fig_GradCAM_random_seed42.png'
if os.path.exists(SRC):
    im=Image.open(SRC).convert('RGB')
    xb=[(25,734),(805,1514),(1585,2294)]; yb=[(84,776),(854,1563),(1641,2349),(2428,3136),(3214,3922),(4001,4709),(4787,5495),(5574,6282)]
    lab=[('Pituitary','Pituitary','Pituitary','Pituitary'),('Glioma','Pituitary','Pituitary','Pituitary'),('Glioma','Glioma','Glioma','Glioma'),('Meningioma','Meningioma','Meningioma','Meningioma'),('Meningioma','Meningioma','Meningioma','Meningioma'),('Meningioma','Glioma','Glioma','Glioma'),('Glioma','Glioma','Glioma','Glioma'),('No Tumor','No Tumor','No Tumor','No Tumor')]
    fig,axs=plt.subplots(4,6,figsize=(18,12.6))
    for k in range(8):
        r=k//2; c0=(k%2)*3; t,f,pr_,ps=lab[k]
        for j in range(3):
            a=axs[r,c0+j]; a.imshow(im.crop((xb[j][0],yb[k][0],xb[j][1],yb[k][1]))); a.axis('off')
            a.set_title([f'({k+1}) True: {t} | fusion: {f}',f'ResNet101 layer4 (pred {pr_})',f'Swin-V2-B last stage (pred {ps})'][j],fontsize=10)
    plt.tight_layout(); plt.savefig(O+'fig10a_gradcam_random.png',dpi=130); plt.close()
