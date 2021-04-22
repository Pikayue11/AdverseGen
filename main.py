import PySimpleGUI as sg
import util_gui as gui
import numpy as np
import L0_attack.L0_API as l0

N = 1
path = ''
imageArr = np.zeros((N,32,32,3))

#   button
Check_image = sg.Button('Check image', key='-ci-')
Comfirm_all = sg.Button('Comfirm all', key='-ca-')
Quit = sg.Button('Quit', key='-quit-')

#   text
# title = sg.Text('Our tool box',key='-title-',justification='center')
t1 = sg.Text('Please select network and database',key='-t1-')
t2 = sg.Text('Please choose a jpg/jpeg/png picture in your computer',key='-t2-')
# t3 = sg.Text('Max queries',key='-t3-')
# t4 = sg.Text('recommend: 100000-200000',key='-t4-')
t5 = sg.Text('progress bar',key='-t5-')


#   Combo
database = sg.Combo(['CIFAR-10', 'ImageNet'],default_value='CIFAR-10',key='-db-')
network = sg.Combo(['Resnet18','AlexNet','VGG16'],default_value='Resnet18',key='-nw-')
evaluation = sg.Combo(['L0','L2','L∞','SSIM'],default_value='L0',key='-ev-')
attackMode = sg.Combo(['NonTarget','Target'],default_value='NonTarget',key='-am-')

#   image select
i1 = sg.Input(key='-ImagePath-')
f1 = sg.FileBrowse('choose picture')

#   image
p1 = sg.Image(key='-ori_image-')
p2 = sg.Image(key='-adv_image-')

#   input
queryLimit = sg.InputText(size=(10,5),key='ql')
#   output
s1 = sg.Output(size=(60, 5))

#   ProgressBar
pb = sg.ProgressBar(1000, orientation='h', size=(45, 10), key='progressbar')

left_column = [
               # [title],
               [t1],
               [database,network,evaluation,attackMode],
               [t2],
               [i1,f1],
               # [t3,queryLimit],
               [Check_image,Comfirm_all,Quit],
               [s1],
               [t5],
               [pb]]

right_column = [[p1],
                [p2]]

layout1 = [[sg.Column(left_column),
            sg.VSeperator(),
            sg.Column(right_column)]]

# window1 = sg.Window('Window 1', layout1, size=(800,600))
window1 = sg.Window('Our tool box', layout1)
win2_active = False

while True:
    event1, values1 = window1.read(timeout=100)

    if event1 in (None, '-quit-'):  # click quit
        break

    if not win2_active and event1 == '-ca-':    # click comfirm all
        print('here is Comfirm all xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    if not win2_active and event1 == '-ci-':    # click check image
        path = window1['-ImagePath-'].get()
        if(path == ''):
            print('path is empty')
        else:
            II = gui.ImageInfo(window1['-db-'].get())
            ori_newPath = gui.savePng(path, II.width, II.height)
            if ori_newPath == '':
                print('Warning! Please enter a correct image path!')
            else:
                print('checked, the label of this image is: tiger')
                window1['-ori_image-'].update(size=(II.width, II.height), filename=ori_newPath)
                window1['-adv_image-'].update(size=(II.width, II.height), filename=ori_newPath)


window1.close()

