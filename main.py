import PySimpleGUI as sg
from PIL import Image
import os

width = 400
height = 300

def savePng(path):
    folder_name = 'pngs/'
    file_name = os.path.basename(path)
    if file_name[-3:].lower() == 'jpg':
        file_name = file_name[:-3]
    if file_name[-4:].lower() == 'jpeg':
        file_name = file_name[:-4]
    file_name += 'png'
    im = Image.open(path)
    im = im.resize((width,height))
    newPath = folder_name + file_name
    im.save(newPath)
    return newPath

#   button
Check_image = sg.Button('Check image', key='-ci-')
Comfirm_all = sg.Button('Comfirm all', key='-ca-')
Quit = sg.Button('Quit', key='-quit-')
##############
w2_b1 = sg.Button('w2_b1',key = 'w2-b1')
#   text
title = sg.Text('Our tool box',key='-title-',justification='center')
t1 = sg.Text('Please select xxxxxxxxxxx',key='-t1-')
t2 = sg.Text('Please choose a jpg/jpeg/png picture',key='-t2-')
t3 = sg.Text('Max queries',key='-t3-')
t4 = sg.Text('recommend: 100000-200000',key='-t4-')
t5 = sg.Text('progress bar',key='-t5-')
#   Combo
database = sg.Combo(['ImageNet','CIFAR-10'],default_value='ImageNet',key='-db-')
network = sg.Combo(['AlexNet','VGG16'],default_value='AlexNet',key='-nw-')
evaluation = sg.Combo(['L0','L2','L∞','SSIM'],default_value='L0',key='-ev-')
attackMode = sg.Combo(['NonTarget','Target'],default_value='NonTarget',key='-am-')
#   image select
i1 = sg.Input(key='-path-')
f1 = sg.FileBrowse('choose picture')
#   image
p1 = sg.Image(size=(width,height),key='-image1-')
p2 = sg.Image(size=(width,height),key='-image2-')
# input
queryLimit = sg.InputText(size=(10,5),key='ql')
#   output
s1 = sg.Output(size=(60, 5))
# , enter_submits=True
#   progressBar 进度条
pb = sg.ProgressBar(1000, orientation='h', size=(40, 10), key='progressbar')


left_column = [[title],
               [t1],
               [database,network,evaluation,attackMode],
                [t2],
               [i1,f1],
               [t3,queryLimit,t4],
               [Check_image,Comfirm_all,Quit],
               [s1],
               [t5],
               [pb]]
right_column = [[p1],[p2]]
layout1 = [[sg.Column(left_column),
            sg.VSeperator(),
            sg.Column(right_column)]]

# window1 = sg.Window('Window 1', layout1,size=(800,600))

window1 = sg.Window('Our tool box', layout1)
win2_active = False



while True:
    event1, values1 = window1.read(timeout=100)

    if event1 in (None, '-quit-'):
        break

    if not win2_active and event1 == '-ca-':
        print('here is Comfirm all')


    if not win2_active and event1 == '-ci-':
        path = i1.get()
        print('here is check image')
        if(path == ''):
            print('path is empty')
        else:
            newPath = savePng(path)
            window1['-image1-'].update(filename=newPath)
            window1['-image2-'].update(filename=newPath)






    # if not win2_active and event1 == 'b4':
    #     win2_active = True
    #     layout2 = [[sg.Text('Warning please xxxxxxxxxxxxxx')],
    #                [sg.Button('w2_b1')]]
    #
    #     window2 = sg.Window('Warning', layout2)
    #
    # if win2_active:
    #     events2, values2 = window2.Read(timeout=100)
    #     if events2 is None or events2 == 'w2_b1':
    #         win2_active = False
    #         window2.close()

window1.close()