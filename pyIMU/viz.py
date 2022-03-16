# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 14:19:54 2022

@author: cordillet_s
"""
import plotly.graph_objects as go
import numpy as np


def labFrame():
    
    fig=go.Figure()

    fig.update_layout(
        showlegend=False,
        scene_camera=dict(
            eye=dict(x=0.1, y=0.8,z=0.1),
            up=dict(x=0,y=0,z=1),
            center=dict(x=0,y=0,z=-0.5)
            ),
        scene = dict(
            xaxis = dict(nticks=10, range=[-4,4],showbackground=False,),
            yaxis = dict(nticks=10, range=[-4,4],showbackground=False,),
            zaxis = dict(nticks=4, range=[-0.05,8-0.05],),
            aspectratio=dict(x=1,y=1,z=1)
            )
        )
    return(fig)

def addOrigin(fig=go.Figure(),
              colors=['lightblue','lightpink','lightgreen'],
              linewidth=15,
              lineLength=0.4, 
              annotations=True
              ):
    fig=addFrame(fig,
                 posInit=[0,0,0],
                 R=[[1,0,0],
                     [0,1,0],
                     [0,0,1]],
                 name='LAB',
                 annotations=annotations
                 )
    
    return(fig)

def addFrame(fig=go.Figure(),
             posInit=[0,0,0],
             R=[[1,0,0],
                [0,1,0],
                [0,0,1]],
             colors=['lightblue','lightpink','lightgreen'],
             linewidth=15,
             lineLength=0.4, 
             annotations=True,
             name='Frame1'
             ):

    R = np.array(R, dtype=float)*lineLength
    
    Frame=[
        go.Scatter3d(
            name= name+'_X',
            x=[posInit[0],posInit[0]+R[0][0]],
            y=[posInit[1],posInit[1]+R[1][0]],
            z=[posInit[2],posInit[2]+R[2][0]],
            line=dict(color =colors[0],width=linewidth),
            mode="lines"
            ),
        go.Scatter3d(
            name=name+'_Y',
            x=[posInit[0],posInit[0]+R[0][1]],
            y=[posInit[1],posInit[1]+R[1][1]],
            z=[posInit[2],posInit[2]+R[2][1]],
            line=dict(color =colors[1],width=linewidth),
            mode="lines"
            ),
        go.Scatter3d(
            name=name+'_Z',
            x=[posInit[0],posInit[0]+R[0][2]],
            y=[posInit[1],posInit[1]+R[1][2]],
            z=[posInit[2],posInit[2]+R[2][2]],
            line=dict(color =colors[2],width=linewidth),
            mode="lines"
            ),
        ]
    for axe in Frame:
        fig.add_trace(axe)
    
    # Ajout des annotations
    if annotations:
        # annotation X
        addAnnotation(fig, 
                      pos=[
                          posInit[0]+R[0][0],
                          posInit[1]+R[1][0],
                          posInit[2]+R[2][0],
                          ],
                      texte='X', 
                      name=name+'_X')
        # annotation Y
        addAnnotation(fig, 
                      pos=[
                          posInit[0]+R[0][1],
                          posInit[1]+R[1][1],
                          posInit[2]+R[2][1],
                          ],
                      texte='Y', 
                      name=name+'_Y')
        # annotation Z
        addAnnotation(fig, 
                      pos=[
                          posInit[0]+R[0][2],
                          posInit[1]+R[1][2],
                          posInit[2]+R[2][2],
                          ],
                      texte='Z', 
                      name=name+'_Z')
    
    return(fig)

def getAnnotation(fig):
    return(list(fig.layout.scene.annotations))

def addAnnotation(fig,
                  pos=[1,0,0],
                  texte='X',
                  visibleArrow=False,
                  name='name'):
    l=getAnnotation(fig)
    if not(l):
        l=[go.layout.scene.Annotation(
            dict(
                showarrow=visibleArrow,
                x=pos[0],
                y=pos[1],
                z=pos[2],
                text=texte,
                name=name
                ))
            ]
    else :
        l.append( 
           go.layout.scene.Annotation(
               dict(
                   showarrow=visibleArrow,
                   x=pos[0],
                   y=pos[1],
                   z=pos[2],
                   text=texte,
                   name=name
                   ))
           )
        
    # print(l) # DEBUG
    fig.layout.scene.annotations=l
    return(fig)


def addFrames(fig,
              pos, 
              r, 
              linewidth=15,
              lineLength=0.4,
              colors=['lightblue','lightpink','lightgreen'],
              ):
        frames=[]
        
        for i, p in zip(r, pos):
            
            
            #DEBUG
            # print('Rotation matrix =====')
            # print(i.as_matrix())
            # print('position array ======')
            # print(p)
            
            R = np.array(r, dtype=float)*lineLength
            frames.append(go.Frame(
                data=[
                    go.Scatter3d(
                        x=[p[0],p[0]+R[0][0]],
                        y=[p[1],p[1]+R[1][0]],
                        z=[p[2],p[2]+R[2][0]],
                        mode="lines",
                        line=dict(color =colors[0],width=linewidth),
                        ),
                    go.Scatter3d(
                        x=[p[0],p[0]+R[0][1]],
                        y=[p[1],p[1]+R[1][1]],
                        z=[p[2],p[2]+R[2][1]],
                        mode="lines",
                        line=dict(color =colors[1],width=linewidth),
                        ),
                    go.Scatter3d(
                        x=[p[0],p[0]+R[0][2]],
                        y=[p[1],p[1]+R[1][2]],
                        z=[p[2],p[2]+R[2][2]],
                        mode="lines",
                        line=dict(color =colors[2],width=linewidth),
                        ),
                    ]
                ))
        fig.frames=frames
        
        sliders = [dict(steps= [dict(method= 'animate',
                                args= [[ frames[k] ],
                                      dict(mode= 'immediate',
                                      frame= dict( duration=600, redraw= True ),
                                                transition=dict( duration= 100)
                                              )
                                        ],
                                  ) for k in range(0,len(frames))], 
                    transition= dict(duration= 200 ),
                    x=0,
                    y=0,
                    currentvalue=dict(font=dict(size=12), visible=True, xanchor= 'center'),
                    len=1.0)
                ]

        fig.update_layout(
            sliders=sliders)
        return(fig)
    
#### ========= > A TERMINER 

def removeFrame(fig, frame_id): #NOT WORKING FOR NOW

    ## Recuperer les objects à modifier
    data=list(fig.data)
    annotations=list(fig.layout.scene.annotations)
    

    #recherche et supprime les élements qui contiennent le nom specifique   
    for g in data:
        print(g.name)
        if frame_id in g.name:
            data.remove(g)
            
    for note in annotations:
        print(note.name)
        if frame_id in note.name:
            annotations.remove(note)
    
    #maj de la figure
    fig.data=data
    fig.layout.scene.annotations=annotations
    
    return(fig)