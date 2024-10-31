import winsound

def successful_run_sound():
    duration = 100  # milliseconds
    freq = 440  # Hz (A4 note)
    winsound.Beep(freq, duration)
    winsound.Beep(freq * 2, duration)  
    winsound.Beep(freq-50, duration)
    winsound.Beep((freq-50) * 3, duration) 
    winsound.Beep(freq, duration)
    winsound.Beep(freq * 2, duration)  
    winsound.Beep(freq-50, duration)
    winsound.Beep((freq-50) * 1, duration) 
    winsound.Beep(freq, duration)
    winsound.Beep(freq * 2, duration)  
    winsound.Beep(freq-50, duration)
    winsound.Beep((freq-50) * 3, duration) 

def error_sound():
    frequency = 440  
    duration = 200  
    winsound.Beep(frequency, duration)
    winsound.Beep(frequency, duration)
    winsound.Beep(frequency, duration)
    winsound.Beep(frequency, duration)
    winsound.Beep(frequency, duration)
    winsound.Beep(frequency, duration*5)
 

def warning_sound():
    frequency = 440  
    duration = 200  
    winsound.Beep(frequency, duration)
    winsound.Beep(frequency, duration)

