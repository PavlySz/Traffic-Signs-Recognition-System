import sys

def tsdrs_main():
    mode = str(input("Welcome to the Traffic Signs Detection and Recognition System. \
                    \n\tSelect a mode: \
                    \n\t[1] Real time\t\t[2] Video input \
                    \n\t[3] Static images\t[4] 1-D frame (sockets) \
                    \n\tPress [0] to exit \
                    \nInput: "))

    if mode == '0':
        print("Goodbye.")
        sys.exit(0)

    elif mode == '1':
        print("[INFO] You entered the Real-Time mode.")
        from TSDRS_Real_Time import tsdrs_rt_main
        tsdrs_rt_main()

    elif mode == '2':
        print("[INFO] You entered the Video Input mode.")
        from TSDRS_video import tsdrs_video_main
        tsdrs_video_main()

    elif mode == '3':
        print("[INFO] You entered the Static Images mode.")
        from TSDRS_Static_Images import tsdrs_static_images_main
        tsdrs_static_images_main()

    elif mode == '4':
        print("[INFO] You entered the 1-D Frame (Sockets) mode.")
        from TSDRS_Frame import tsdrs_frame_main
        tsdrs_frame_main()

    else:
        print("Invalid input. Please enter a number between (0 and 4)!")
        tsdrs_main()


if __name__ == '__main__':
    tsdrs_main()