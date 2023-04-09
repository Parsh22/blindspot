from selenium import webdriver
from selenium.webdriver.common.by import By


def browser_call():
    # Set the remote debugging port to connect to an existing Chrome instance
    options = webdriver.ChromeOptions()
    permissions = {
        "profile.default_content_setting_values.media_stream_mic": 1,
        "profile.default_content_setting_values.media_stream_camera": 1
    }
    options.add_experimental_option("prefs", permissions)
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument('--remote-debugging-port=9222')

    # Create a new instance of the Chrome driver
    driver = webdriver.Chrome(options=options)

    # Open a new tab in the existing Chrome window
    driver.execute_script("window.open('about:blank','tab2');")

    # Navigate to the website in the new tab
    link = 'https://videosdk.live/prebuilt/demo'
    driver.switch_to.window("tab2")
    driver.get(link)

    # Refresh the page
    driver.refresh()
    driver.implicitly_wait(20)  # gives an implicit wait for 20 seconds

    iframe = driver.find_element(By.XPATH, "/html/body/div[2]/iframe")
    driver.switch_to.frame(iframe)

    # find the element inside the iframe
    element = driver.find_element(By.ID, "inputJoin")
    element.send_keys("Help Seeker")

    button_element = driver.find_element(By.ID, "btnJoin")
    button_element.click()

    driver.implicitly_wait(20)
    on_webcam_element = driver.find_element(By.ID, "btnWebcam")
    on_webcam_element.click()
    on_mic_element = driver.find_element(By.ID, "btnMic")
    on_mic_element.click()

    # Keep the script running and wait for the user to press Enter
    input("Press Enter to close the browser...")
    driver.quit()
