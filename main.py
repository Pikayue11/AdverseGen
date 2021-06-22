import PySimpleGUI as sg
import util_gui as gui
import threading
import Thread_control as tc
from backEnd.attacker import ImageAttacker
import numpy as np

# Initialize First place
database_arr = ['CIFAR-10', 'ImageNet']
cons_number = 2
cons_max = 4
cons_list = ['Changed pixels', 'Largest pixel difference']
cons_reverse = ['Structure similarity' ,'Euclidean distance']

# Window
#   button
run_attack = sg.Button('Run attack', key='-ra-')
Stop = sg.Button('Stop', key='-stop-')
Quit = sg.Button('Quit', key='-quit-')

Console = sg.Button('Console', key='-console-')
Add_Cons = sg.Button('Add', key='-add-') # add more constraints, open second window
c1_del = sg.Button('Delete', pad=(44,0), key='-c1_del-')
c2_del = sg.Button('Delete', pad=(44,0), key='-c2_del-')
c3_del = sg.Button('Delete', pad=(44,0), key='-c3_del-')
c4_del = sg.Button('Delete', pad=(44,0), key='-c4_del-')

#   text
t1 = sg.Text('Please select database, network model and attack mode', key='-t1-')
t2 = sg.Text('Please choose a jpg/jpeg/png picture in your computer', key='-t2-')
t3 = sg.Text('States: free          ', font='Arial 12', pad=(180,10), key='-t3-')
t4 = sg.Text('Please select the attack norm and constraints', key='-t4-')
ori_label = sg.Text('', size=(25, 1), key='-ori_label', justification='center')
pert_value = sg.Text('', size=(25, 1), key='-pert_value-', justification='center')
adv_label = sg.Text('', size=(25, 1), key='-adv_label-', justification='center')
limit = sg.Text('Constraints', size=(10, 1), font='Arial 20', key='-limit-')
limit_value = sg.Text('Value', size=(6, 1), font='Arial 20', key='-limit_value-')
c1 = sg.Text('Changed pixels', size=(22, 1), key='-c1-')
c2 = sg.Text('Largest pixel difference', size=(22, 1), key='-c2-')
c3 = sg.Text('', size=(22, 1), key='-c3-')
c4 = sg.Text('', size=(22, 1), key='-c4-')

#   Combo
evaluation = sg.Combo(['L0', 'L2', 'L∞', 'SSIM', 'Decision-based'], readonly=True, default_value='SSIM', key='-ev-')
attackMode = sg.Combo(['Target', 'NonTarget'], default_value='Target', readonly=True, change_submits=True, key='-am1-')
attackMode2 = sg.Combo(['Target', 'NonTarget'], default_value='Target', readonly=True, change_submits=True, key='-am2-')
base = sg.Combo(['Score based', 'Decision based'], default_value='Score based', readonly=True, change_submits=True, key='-ba-')

#   Combo initailize:
II = gui.ImageInfo(database_arr[0])
imageAttacker = ImageAttacker(database_arr[0])
arr1 = imageAttacker.get_available_model()
arr2 = II.getLabelArray()
network = sg.Combo(values=arr1, default_value=arr1[0], readonly=True, key='-nw1-') # in table 1
labelArr = sg.Combo(values=arr2,default_value=arr2[0], readonly=True, change_submits=True, key='-la1-') # in table 1
II.set_database(database_arr[1])
imageAttacker.set_database(database_arr[1])
arr1 = imageAttacker.get_available_model()
arr2 = II.getLabelArray()
network2 = sg.Combo(values=arr1, readonly=True, default_value=arr1[0], key='-nw2-') # in table 2
labelArr2 = sg.Combo(values=arr2, readonly=True, default_value=arr2[0], change_submits=True, key='-la2-') # in table 2

#   FileBrowse
f1 = sg.FileBrowse( 'choose picture as target', change_submits=True, key='-cp1-')
f2 = sg.FileBrowse('choose picture as target', change_submits=True, key='-cp2-')
f3 = sg.FileBrowse('choose picture to attack', change_submits=True, key='-cp-')

#   image
p1 = sg.Image(size=(100, 200), key='-ori_image-')
p2 = sg.Image(size=(100, 200), key='-pert_image-')
p3 = sg.Image(size=(100, 200), key='-adv_image-')

#   input
c1_value = sg.InputText('', size=(5,1), key='-c1_value-')
c2_value = sg.InputText('', size=(5,1), key='-c2_value-')
c3_value = sg.InputText('', size=(5,1), key='-c3_value-')
c4_value = sg.InputText('', size=(5,1), key='-c4_value-')

# temp ########################################################### for test
testButton = sg.Button('test', key='-test-')
testButton2 = sg.Button('test2', key='-test2-')

#   tables
tab1_layout = [[network, attackMode, labelArr], [f1]] #cifar-10
tab2_layout = [[network2, attackMode2, labelArr2],[f2]] #imageNet
tab1 = sg.Tab('CIFAR-10', tab1_layout, key='CIFAR-10')
tab2 = sg.Tab('ImageNet', tab2_layout, key='ImageNet')
gro = sg.TabGroup([[tab1,tab2]], enable_events=True, key='-group-', tab_background_color='grey', tab_location='lefttop')

#   layout
layout1_1 = [[t1],[gro]]
layout1_2 = [[limit, limit_value, Add_Cons]]
layout1_3 = [[c1, c1_value, c1_del]]
layout1_4 = [[c2, c2_value, c2_del]]
layout1_5 = [[c3, c3_value, c3_del]]
layout1_6 = [[c4, c4_value, c4_del]]
layout1_7 = [[base, f3, run_attack]]
layout1_8 = [[p1, p2, p3], [ori_label, pert_value, adv_label]]
layout1_9 = [[Console, Stop, Quit, t3],
             # [testButton, testButton2],
           ]
layout1 = [[sg.Column(layout1_1, key='-lay1-')],
           [sg.HSeparator()],
           [sg.Column(layout1_2, key='-lay2-')],
           [sg.Column(layout1_3, key='-lay3-')],
           [sg.Column(layout1_4, key='-lay4-')],
           [sg.Column(layout1_5, visible=False, key='-lay5-')],
           [sg.Column(layout1_6, visible=False, key='-lay6-')],
           [sg.Column(layout1_7, key='-lay7-')],
           [sg.HSeparator(),],
           [sg.Column(layout1_8, key='-lay8-')],
           [sg.HSeparator()],
           [sg.Column(layout1_9, key='-lay9-')]
           ]

# Windows
win1 = sg.Window('Our tool box', layout1, size=(700, 630))  # 700, 650

# Initialize second place
path = ''
threads = []
label = 0
win2_active = False
win3_active = False
ori_images = np.array([])
II.set_database(database_arr[0])
imageAttacker.set_database(database_arr[0])

cp_save = ['-cp1-', '-cp2-']
attMode_save = ['-am1-', '-am2-']
label_save = ['-la1-','-la2-']
lay_save = ['-lay3-', '-lay4-', '-lay5-', '-lay6-']
c_save = ['-c1-', '-c2-', '-c3-', '-c4-']
del_save = ['-c1_del-', '-c2_del-', '-c3_del-', '-c4_del-']
aaa = 1
while True:
    event1, values1 = win1.read(timeout=100)

    if event1 in ('-c1_del-', '-c2_del-', '-c3_del-', '-c4_del-') and not win2_active:  # after click one of 'Delete'
        index = del_save.index(event1)
        cons_list.remove(win1[c_save[index]].get())
        cons_reverse.append(win1[c_save[index]].get())
        if index + 1 == cons_number:
            win1[lay_save[index]].update(visible=False)
        if index + 1 < cons_number:
            win1[lay_save[cons_number-1]].update(visible=False)
            for i in range(index, cons_number-1):
                win1[c_save[i]].update(win1[c_save[i+1]].get())
        cons_number -= 1

    if event1 == '-cp1-' and not win2_active:   # after choose a target image in cifar-10, remain work, see reference at bottom
        print(win1['-cp1-'].TKStringVar.get())
    if event1 == '-cp2-' and not win2_active:   # after choose a target image in imageNet, remain work
        print(win1['-cp2-'].TKStringVar.get())
    if event1 == '-cp-' and not win2_active:   # after choose a original image, remain work
        print(win1['-cp-'].TKStringVar.get())

    if event1 == '-group-' and not win2_active:    # select database in table group, set database in II and image Attacker , win1['-group-'].get() returns 'CIFAR-10' or 'ImageNet'
        II.set_database(win1['-group-'].get())
        imageAttacker.set_database(win1['-group-'].get())

    if event1 in ('-am1-', '-am2-') and not win2_active:  # select attack mode in either of table group, it will disable or enable some components
        index = attMode_save.index(event1)
        if win1[event1].get() == 'Target':
            win1[label_save[index]].update(visible=True)
            win1[cp_save[index]].update(disabled=False)
        else:
            win1[label_save[index]].update(visible=False)
            win1[cp_save[index]].update(disabled=True)

    if event1 in (None, '-quit-') and not win2_active:  # click quit, stop all the thread and break
        for i in threads:
            tc._async_raise(i.ident, SystemExit)
        break

    if event1 in (None, '-stop-') and not win2_active:  # click stop, stop all the thread and empty threads array
        for i in threads:
            tc._async_raise(i.ident, SystemExit)
        threads = []
        win1['-t6-'].update('States: free          ')
        win1['-ori_image-'].update()
        win1['-adv_image-'].update()
        print('stopped')

    if event1 == '-ra-' and not win2_active:  # click run attack ,remain work
        if win1['-t6-'].get()[8:12] == 'free':
            win1['-t6-'].update('States: running   ')
            t1 = threading.Thread(target=gui.getAdvPath, args=(imageAttacker, ori_images, label, II, win1,))
            threads.append(t1)
            t1.start()
            t2 = threading.Thread(target=gui.updateRunning, args=(win1,))
            threads.append(t2)
            t2.start()
        else:
            print('There is something running, please wait')

    if event1 == '-add-' and not win2_active:    # click edit constraints, open window 2
        if cons_max > cons_number:
            win2_active = True
            layout2 = [[sg.Text('Add new constraints')],
                       [sg.Combo(cons_reverse, readonly=True, default_value=cons_reverse[0], key='-cr-')],
                       [sg.Button('ok', key='-ok2-'), sg.Button('cancel', key='-cancel2-')]]
            win2 = sg.Window('Constraints', layout2)
        else:
            sg.popup('You have added all the constraints', title='warning')

    if win2_active:
        event2, valus2 = win2.read(timeout=100)

        if event2 == '-ok2-':   # click ok button, add a new constraint
            str = win2['-cr-'].get()
            cons_list.append(str)
            cons_reverse.remove(str)
            win1[c_save[cons_number]].update(str)
            win1[lay_save[cons_number]].update(visible=True)
            cons_number += 1
            win2.close()
            win2_active = False

        if event2 in (None, '-cancel2-'):   #   click cancel button, do nothing
            win2.close()
            win2_active = False

    if event1 == '-console-' and not win3_active:   # click to open or close the console, working on it
        win3_active = True
        layout3 = [[sg.Output(size=(100, 10), key='-output-', echo_stdout_stderr=True)], [sg.Button('Clear', key='-clear3-'), sg.Button('Close', key='-close3-')]]
        win3 = sg.Window('Console', layout3)

    if win3_active:
        event3, valus3 = win3.read(timeout=100)

        if event3 == '-clear3-':
            win3['-output-'].update('')

        if event3 in (None, '-close3-'):   #  close
            win3.close()
            win3_active = False

win1.close()



# reference ###############################################################


# if event1 == '-test-':
#     pass
#
# if event1 == '-test2-':
#     pass


# if event1 == '-ci-' and not win2_active:  # click check image
#     path = win1['-ImagePath-'].get()
#     if path == '':
#         print('path is empty')
#     else:
#         II = gui.ImageInfo(win1['-group-'].get())
#         ori_newPath = gui.savePng(path, II.width, II.height)
#         if ori_newPath == '':
#             print('Warning! Please enter a correct image path!')
#         else:
#             ori_images = gui.getImage(ori_newPath)
#             ori_label_id = imageAttacker.get_label(win1['-nw1-'].get(), ori_images / 255)
#             label = ori_label_id
#             ori_label_name = II.mapLabel(ori_label_id)
#             print("The label of the original image is", ori_label_name)
#             ori_image_zoom = gui.convert_to_bytes(ori_newPath, (200, 200))
#             win1['-ori_image-'].update(data=ori_image_zoom)
#             win1['-ori_label'].update(f'label: {ori_label_name}')


# if event1 == '-ImagePath-' and not win2_active:  # the common part, path of original image
#     ori_new_path = win1['-ImagePath-'].get()
#     ori_image = gui.getImage(ori_new_path)
#     ori_image_zoom = gui.convert_to_bytes(ori_new_path, (200, 200))
#     p1.update(data=ori_image_zoom)
