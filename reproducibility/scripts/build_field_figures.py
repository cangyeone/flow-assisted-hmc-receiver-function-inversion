"""Redraw the completed field cohort from frozen arrays; no inversion or interpolation.

Run from any directory. Uses only NumPy/Matplotlib and the archived manuscript
data. The map projection is equirectangular at 36 N. A--A' is selected solely
by |longitude + 108.5| <= 0.22 degrees, before looking at profile outcomes.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "field-figures"
INK = '#22323D'
BLUE = '#176B8E'
ORANGE = '#C67735'
GRAY = '#85939B'
SAND = '#EFE8DB'
VSCMAP = 'RdYlBu'
VSNORM = Normalize(.5, 4.5)
INPUTS = set()


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 7.5,
        'axes.labelsize': 7.5, 'axes.titlesize': 8.2,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': .55, 'lines.linewidth': 1,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.major.width': .5, 'ytick.major.width': .5,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'svg.hashsalt': 'field-figures-20261011',
    })


def read(path, arrays=False):
    path = ROOT / path
    INPUTS.add(path)
    return dict(np.load(path)) if arrays else json.loads(path.read_text())


def save(fig, name):
    for suffix in ['pdf', 'svg', 'png']:
        destination=FIGURES / f'{name}.{suffix}'
        fig.savefig(destination, dpi=300)
        if suffix == 'svg':
            destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
    plt.close(fig)


def title(ax, letter, text):
    ax.set_title(f'{letter}  {text}', loc='left', pad=7, fontweight='medium')


def polygon(ax, geom, strong=False):
    polygons = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
    for poly in polygons:
        vertices, codes = [], []
        for ring in poly:
            vertices.extend(ring)
            codes.extend([MPath.MOVETO] + [MPath.LINETO] * (len(ring)-2) + [MPath.CLOSEPOLY])
        ax.add_patch(PathPatch(MPath(vertices, codes),
                              facecolor=SAND if strong else '#F7F7F3',
                              edgecolor='#BBAD97' if strong else '#D8DCD8',
                              lw=.55 if strong else .3, zorder=0))


def basemap(ax, geo, labels=False, transect=False):
    for f in geo['features']:
        polygon(ax, f['geometry'], f['properties']['Name'] in
                ['San Juan Basin', 'Zuni Basin', 'Paradox Basin'])
    ax.set(xlim=(-110, -103), ylim=(33, 39), xticks=[-110, -108, -106, -104], yticks=[33, 35, 37, 39])
    ax.set_aspect(1 / np.cos(np.deg2rad(36)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{abs(v):.0f}°W'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}°N'))
    ax.grid(color='#E2E6E7', lw=.35, zorder=-2)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(True)
    if labels:
        for name, x, y in [('Paradox\nBasin', -109.35, 38.68), ('San Juan', -107.9, 36.95),
                           ('Zuni', -108.8, 34.08), ('Rio Grande\nrift basins', -106.0, 37.8)]:
            ax.text(x, y, name, fontsize=6.5, color='#816B4D', ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', pad=.4, alpha=.7), zorder=4)
    if transect:
        ax.axvspan(-108.72, -108.28, color=BLUE, alpha=.055, zorder=1)
        ax.plot([-108.5, -108.5], [33.0, 38.6], color=INK, lw=.65, ls=(0,(4,3)), zorder=2)
        for name, y in [('A', 38.65), ("A′", 33.06)]:
            ax.text(-108.35, y, name, fontsize=8, weight='bold', va='bottom', zorder=5)


def load_cohort():
    context = read('data/basin-assessment/context/station-context.json')
    records = []
    for c in context:
        s = c['station']
        d = read(f'data/results/{s}-hmc.json')
        a = read(f'data/basin-assessment/real/{s}.npz', True)
        result = read(f'data/basin-assessment/real/{s}.json')
        candidates = read(f'data/results/{s}-map.json')['candidates']
        baseline = min(candidates, key=lambda x: x['negative_log_density'])['x']
        lo, hi = np.array(d['bounds']); u = (np.array(d['mean'])-lo)/(hi-lo)
        mask = (a['time'] >= .799) & (a['time'] <= 26.001)
        scores = {}
        for key in ['compact', 'selected']:
            scores[key] = float(np.sqrt(np.mean((a[key+'_test'][:,mask]-a['rf_test'][:,mask])**2)))
            np.testing.assert_allclose(scores[key], result[key+'_test']['rmse'], atol=2e-12, rtol=0)
        selected = next(x for x in result['checks'] if x['key'] == result['selected'])
        slo, shi = np.array(selected['bounds'])
        su = (np.array(selected['full_candidates'][0]['x'])-slo)/(shi-slo)
        layers = a['layers'][:,:3]
        assert np.all(np.diff(layers[:,0]) > 0)
        records.append(dict(**c, mean=d['mean'], q025=d['q025'], q975=d['q975'],
                            passed=bool(d['diagnostics']['converged'] and not sum(d['production_divergences'])),
                            near_bound=bool(np.any((u<.01)|(u>.99))),
                            selected_near_bound=bool(np.any((su<.01)|(su>.99))),
                            baseline_map_moho=float(baseline[2]), selected_moho=float(layers[-1,0]),
                            selected_layers=layers.tolist(), arrays=a, scores=scores))
    assert len(records) == 42
    return records


def velocity(r, depth):
    a = np.array(r['selected_layers'])
    return a[np.searchsorted(a[:,0], depth, side='right')-1,2]


def regional(records, geo):
    """Geographic overview and a genuinely sampled, non-interpolated corridor."""
    section = sorted([r for r in records if abs(r['longitude']+108.5) <= .22], key=lambda r:-r['latitude'])
    assert [r['station'] for r in section] == ['R20A', 'S20A', 'U20A', 'V20A', 'X20A', 'Z20A']
    lon = [r['longitude'] for r in records]; lat = [r['latitude'] for r in records]
    fig = plt.figure(figsize=(180/25.4, 190/25.4))
    gs = fig.add_gridspec(3,2,left=.09,right=.975,bottom=.13,top=.945,
                          height_ratios=[1.7,.70,1.05],hspace=.56,wspace=.30)
    for col, vals, cmap, norm, label, heading in [
        (0,[velocity(r,.5) for r in records],VSCMAP,VSNORM,'$V_S$ (km s$^{-1}$)','Shallow structure · 0.5 km'),
        (1,[r['selected_moho'] for r in records],'viridis_r',Normalize(15,75),'Selected MAP Moho (km)','Moho depth · all 42 stations')]:
        ax=fig.add_subplot(gs[0,col]);basemap(ax,geo,labels=(col==0),transect=True)
        sc=ax.scatter(lon,lat,c=vals,cmap=cmap,norm=norm,s=33,edgecolors=INK,linewidths=.4,zorder=3)
        if col==1:
            bad=[r for r in records if not r['passed']]
            ax.scatter([r['longitude'] for r in bad],[r['latitude'] for r in bad],marker='x',s=32,c='white',lw=.8,zorder=4)
        for s, off in [('V20A',(5,-7)),('X20A',(5,4))]:
            r=next(r for r in records if r['station']==s)
            ax.scatter(r['longitude'],r['latitude'],s=65,facecolors='none',edgecolors=INK,lw=.6,zorder=5)
            ax.annotate(s,(r['longitude'],r['latitude']),xytext=off,textcoords='offset points',fontsize=6.5,
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.5),zorder=6)
        cb=fig.colorbar(sc,ax=ax,orientation='horizontal',pad=.095,fraction=.044,aspect=27)
        cb.set_label(label,labelpad=2);cb.ax.tick_params(labelsize=6.5)
        title(ax,'ab'[col],heading)
    for row,zmax,letter,heading in [(1,8,'c',"A–A′  Shallow structure"),(2,80,'d',"A–A′  Crustal structure")]:
        ax=fig.add_subplot(gs[row,:]);ax.set_facecolor('#FCFCFA')
        for r in section:
            x=(38.6-r['latitude'])*111.195
            layers=np.array(r['selected_layers'])
            for j,layer in enumerate(layers):
                z0=layer[0];z1=min(layers[j+1,0] if j+1<len(layers) else zmax,zmax)
                if z0>=zmax:break
                ax.add_patch(Rectangle((x-12.5,z0),25,z1-z0,facecolor=plt.get_cmap(VSCMAP)(VSNORM(layer[2])),edgecolor='none'))
            ax.plot([x-12.5,x-12.5,x+12.5,x+12.5],[zmax,0,0,zmax],color='#B8BEC0',lw=.3)
            if row==1:
                ax.text(x,-.8,r['station'],ha='center',fontsize=6.8,
                        fontweight='bold' if r['station'] in ['V20A','X20A'] else 'normal',clip_on=False)
            if row==2:
                ax.scatter(x,r['selected_moho'],marker='v',s=28,c=INK,edgecolors='white',lw=.35,zorder=6)
                ax.scatter(x,r['baseline_map_moho'],marker='D',s=21,facecolors='white',edgecolors=ORANGE,lw=.85,zorder=7)
        ax.set(xlim=(0,630),ylim=(zmax,0),ylabel='Depth (km)',xticks=np.arange(0,601,100))
        ax.set_yticks([0,4,8] if row==1 else [0,20,40,60,80])
        ax.grid(axis='y',color='#DCE1E1',lw=.35,zorder=-1);ax.set_axisbelow(True)
        title(ax,letter,heading)
        if row==1:
            ax.set_title(f'{letter}  {heading}',loc='left',y=1.30,pad=0,fontweight='medium')
            ax.tick_params(labelbottom=False)
            ax.text(.995,1.30,'6 stations · 25 km display widths',transform=ax.transAxes,ha='right',fontsize=6.7,color=GRAY)
        else:
            ax.set_xlabel('Projected distance south from A (km)',labelpad=3)
            ax.text(0,-.31,'A  North',transform=ax.transAxes,fontsize=7)
            ax.text(1,-.31,'South  A′',transform=ax.transAxes,fontsize=7,ha='right')
            fig.legend([Line2D([],[],marker='D',ls='',mfc='white',mec=ORANGE,ms=4),
                       Line2D([],[],marker='v',ls='',color=INK,ms=4)],
                      ['Original MAP Moho','Selected MAP Moho'],loc='center',bbox_to_anchor=(.56,.063),
                      ncol=2,frameon=False,fontsize=7,handletextpad=.4)
    fig.text(.09,.034,'Independent 1-D models; blank space between stations is unsampled. Depth is below each station.',fontsize=6.8)
    fig.text(.09,.017,'× Unresolved baseline chains (map b). All selected MAP models contain at least one near-bound parameter.',fontsize=6.8,color=GRAY)
    save(fig,'regional-structure')
    return [r['station'] for r in section]


def field_fits(records):
    choices=read('data/basin-assessment/figure-protocol.json')['real_examples']
    manifest={r['station']:r for r in read('data/processed/manifest.json')}
    fig=plt.figure(figsize=(180/25.4,185/25.4))
    outer=fig.add_gridspec(2,1,left=.085,right=.98,bottom=.12,top=.855,hspace=.52)
    handles=[Line2D([],[],color=INK,lw=.85),Line2D([],[],color=ORANGE,lw=1,ls=(0,(4,2))),
             Line2D([],[],color=BLUE,lw=1.1),Patch(facecolor='#DDE3E6',edgecolor='none')]
    fig.legend(handles,['Observed RF','Previous compact MAP','Selected MAP','±1 stack SE'],loc='upper center',
               bbox_to_anchor=(.53,.986),ncol=2,frameon=False,columnspacing=2.7,fontsize=7.4,handlelength=2.7)
    audits=[]
    for row,choice in enumerate(choices):
        s=choice['station'];r=next(r for r in records if r['station']==s);a=r['arrays']
        grid=outer[row].subgridspec(1,3,width_ratios=[.78,.86,3.0],wspace=.52)
        for col,zmax,heading in [(0,8,'Shallow model'),(1,50,'Crustal model')]:
            ax=fig.add_subplot(grid[col])
            for key,color,ls in [('compact',ORANGE,(0,(4,2))),('selected',BLUE,'-')]:
                edges=np.r_[0,(a['depth'][:-1]+a['depth'][1:])/2,50.]
                ax.plot(np.repeat(a[key+'_profile'],2),np.column_stack([edges[:-1],edges[1:]]).ravel(),
                        color=color,ls=ls,lw=1.15)
            ax.set(xlim=(.4,4.7),ylim=(zmax,0),xticks=[1,3] if col==0 else [2,4],
                   ylabel='Depth (km)',xlabel='$V_S$ (km s$^{-1}$)')
            ax.set_yticks([0,2,4,6,8] if col==0 else [0,10,20,30,40,50])
            ax.grid(axis='y',color='#E0E4E5',lw=.4);ax.set_axisbelow(True)
            title(ax,'abcdef'[row*3+col],heading)
            if col==1:
                ax.axhspan(0,8,color='#E9EFF1',zorder=-1)
        gs=grid[2].subgridspec(3,1,hspace=.20)
        for b in range(3):
            ax=fig.add_subplot(gs[b]);t=a['time']
            ax.axvspan(-.6,.8,color='#F0EFEA',zorder=-1)
            ax.axhline(0,color='#B9C1C5',lw=.45)
            ax.fill_between(t,a['rf_test'][b]-a['test_se'][b],a['rf_test'][b]+a['test_se'][b],color='#DDE3E6',lw=0,zorder=1)
            ax.plot(t,a['compact_test'][b],color=ORANGE,ls=(0,(4,2)),lw=1,zorder=2)
            ax.plot(t,a['selected_test'][b],color=BLUE,lw=1.05,zorder=3)
            ax.plot(t,a['rf_test'][b],color=INK,lw=.7,zorder=4)
            ax.set(xlim=(-.6,26),ylim=(-2.9,5.3),xticks=[0,5,10,15,20,25],yticks=[0,3])
            ax.text(.99,.86,f"p = {a['p_test'][b]:.3f} s/km; n = {manifest[s]['test_counts'][b]}",
                    ha='right',va='top',transform=ax.transAxes,fontsize=6.5,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.3))
            if b==0:title(ax,'cf'[row],'Held-out receiver functions')
            if b==1:ax.set_ylabel('Normalized RF',labelpad=1)
            if b<2:ax.tick_params(labelbottom=False)
            else:ax.set_xlabel('Time after P (s)',labelpad=3)
        y=.908 if row==0 else .458
        fig.text(.085,y,f"{s}  /  {choice['basin']}",fontsize=10.0,fontweight='bold',color=INK)
        fig.text(.98,y,f"Held-out RMSE  {r['scores']['compact']:.3f} → {r['scores']['selected']:.3f}",
                 ha='right',fontsize=8,color=BLUE)
        if s=='X20A':fig.text(.45,y,'Baseline unresolved',fontsize=6.3,color=GRAY)
        audits.append(dict(station=s,p=a['p_test'].tolist(),counts=manifest[s]['test_counts'],rmse=r['scores']))
    fig.text(.085,.040,'Identical RF amplitude scales · unsmoothed stacks · three held-out slowness bins per station',fontsize=7)
    fig.text(.085,.019,'MAP/model-adequacy comparison. Grey bands show event-stack variability; they are not posterior intervals.',fontsize=6.8,color=GRAY)
    save(fig,'field-rf-fits')
    return audits


def diagnostics(records,geo):
    fig=plt.figure(figsize=(180/25.4,168/25.4))
    gs=fig.add_gridspec(2,2,left=.10,right=.965,top=.93,bottom=.115,wspace=.42,hspace=.59,height_ratios=[1.25,1])
    lon=np.array([r['longitude'] for r in records]);lat=np.array([r['latitude'] for r in records])
    width=np.array([r['q975'][2]-r['q025'][2] for r in records])
    change=np.array([r['selected_moho']-r['baseline_map_moho'] for r in records])
    ok=np.array([r['passed'] for r in records]);bound=np.array([r['near_bound'] for r in records])
    for col,values,cmap,norm,heading,label in [
        (0,width,'magma_r',LogNorm(float(width.min()),float(width.max())),'Conditional Moho intervals','95% interval width (km; log scale)'),
        (1,change,'RdBu_r',Normalize(-50,50),'Model-assessment sensitivity','Selected − original MAP Moho (km)')]:
        ax=fig.add_subplot(gs[0,col]);basemap(ax,geo)
        sc=ax.scatter(lon,lat,c=values,cmap=cmap,norm=norm,s=33,edgecolors='white',lw=.4,zorder=3)
        ax.scatter(lon[~ok],lat[~ok],marker='x',s=33,c=INK,lw=.8,zorder=4)
        ax.scatter(lon[bound],lat[bound],s=56,facecolors='none',edgecolors=GRAY,lw=.5,zorder=2)
        cb=fig.colorbar(sc,ax=ax,orientation='horizontal',fraction=.045,pad=.10,aspect=28)
        cb.set_label(label,labelpad=2)
        title(ax,'ab'[col],heading)
    ax=fig.add_subplot(gs[1,0]);ax.plot([0,.46],[0,.46],c=GRAY,lw=.7,ls=(0,(3,3)))
    before=np.array([r['scores']['compact'] for r in records]);after=np.array([r['scores']['selected'] for r in records])
    for member,color in [(False,GRAY),(True,BLUE)]:
        g=np.array([bool(r['basins'])==member for r in records])
        ax.scatter(before[g],after[g],s=21,c=color,edgecolors='white',lw=.35,zorder=3,label='Basin polygon' if member else 'Other station')
    for s,off in [('V20A',(-45,6)),('X20A',(-30,8))]:
        i=next(i for i,r in enumerate(records) if r['station']==s)
        ax.scatter(before[i],after[i],s=40,marker='^',color=ORANGE,edgecolors='white',lw=.4,zorder=5)
        ax.annotate(s,(before[i],after[i]),xytext=off,textcoords='offset points',fontsize=6.7)
    ax.set(xlim=(0,.46),ylim=(0,.46),xticks=[0,.2,.4],yticks=[0,.2,.4],xlabel='Previous compact RF RMSE',ylabel='Selected RF RMSE')
    ax.set_aspect('equal');title(ax,'c','All 42 held-out outcomes')
    ax.legend(loc='upper left',frameon=False,fontsize=6.7,handletextpad=.3)
    ax.text(.97,.04,'26 improved\n3 worsened\n13 unchanged',transform=ax.transAxes,ha='right',fontsize=6.8)
    ax=fig.add_subplot(gs[1,1]);absolute=abs(change)
    ax.plot([.01,60],[.01,60],color=GRAY,lw=.7,ls=(0,(3,3)))
    ax.scatter(width,absolute,s=21,c=BLUE,edgecolors='white',lw=.35,zorder=3)
    ax.scatter(width[~ok],absolute[~ok],s=23,marker='x',color=INK,lw=.7,zorder=4)
    ax.set(xscale='log',yscale='log',xlim=(width.min()/1.3,width.max()*1.3),ylim=(absolute.min()/1.5,absolute.max()*1.5),xlabel='Conditional 95% Moho width (km)',ylabel='Absolute MAP Moho change (km)')
    title(ax,'d','Uncertainty versus sensitivity')
    ax.text(.98,.04,f'Medians: {np.median(width):.2f} km vs {np.median(absolute):.1f} km',transform=ax.transAxes,fontsize=6.7,ha='right')
    fig.text(.10,.036,'× Unresolved baseline chains   ○ Baseline mean near a bound   Basin outlines: USGS (2012)',fontsize=6.8)
    fig.text(.10,.017,'Intervals condition on the baseline model; MAP shifts also reflect changes in bounds and residual weighting.',fontsize=6.7,color=GRAY)
    save(fig,'field-diagnostics')
    return dict(stations=len(records),passed=int(ok.sum()),near_bound=int(bound.sum()),
                selected_near_bound=sum(r['selected_near_bound'] for r in records),
                improved=int(np.sum(after<before-1e-12)),worsened=int(np.sum(after>before+1e-12)),
                unchanged=int(np.sum(abs(after-before)<=1e-12)),
                median_conditional_moho_width=float(np.median(width)),median_abs_moho_change=float(np.median(absolute)))


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    style(); records=load_cohort(); geo=read('data/basin-assessment/context/usgs-regional-basins.geojson')
    section=regional(records,geo);examples=field_fits(records);summary=diagnostics(records,geo)
    assert (summary['passed'],summary['near_bound'],summary['selected_near_bound']) == (35,32,42)
    assert (summary['improved'],summary['worsened'],summary['unchanged']) == (26,3,13)
    INPUTS.add(Path(__file__).resolve())
    out=FIGURES
    provenance=dict(summary=summary,examples=examples,section_stations=section,
        section_rule='33 <= latitude <= 39; abs(longitude + 108.5) <= 0.22 degrees; ordered north to south',
        projection='Equirectangular, standard parallel 36 N; projected meridional distance uses 111.195 km/degree from 38.6 N',
        section_display='25 km equal-width station strips, not lateral resolution; no interpolation; depth below each station',
        velocity_map_depth_km=.5,velocity_colour_limits_kms=[.5,4.5],
        profile_display='Archived compact and selected Vs profiles sampled every 0.1 km; steps at sample midpoints',
        waveform_display='No smoothing, renormalization, clipping or time shifting. All six bins share amplitude limits [-2.9,5.3].',
        scope='Layout revision only. Frozen observations, model estimates, example choices and scores remain unchanged.',
        inputs={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(INPUTS)},
        outputs={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                 for name in ['regional-structure','field-rf-fits','field-diagnostics']
                 for f in sorted(FIGURES.glob(name+'.*'))})
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
