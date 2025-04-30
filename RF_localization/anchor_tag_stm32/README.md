# STM32 Anchor and Tag Code

### Setup
1. Clone this repository.
2. Download the STM32CubeIDE (available at https://www.st.com/en/development-tools/stm32cubeide.html).
3. Navigate to `File` -> `Open Projects from File System` -> `Directory`, select this directory (anchor_tag_stm32) in the file system, and click `Finish`.
4. The project should now show up in the `Project Explorer` on the left side of the screen as `NEAR_ANCHOR`. Press `Ctrl+B` or `Project` -> `Build All` to build.

The code for ranging is located in the `Core/` folder. Use the definitions in `Core/Inc/main.h` to determine whether the tag or anchor code is built. The tag and anchor ID's can be configured in `Core/Inc/id.h`. When the STLINK (included with the hardware) is attached to the tag or anchor, press the green play button to flash the device.

### Resources
* This codebase is adapted from the example code available at https://www.decawave.com/wp-content/uploads/2019/01/dw1000_api_rev2p14.zip.
* The DW1000 user manual provides detailed information on using the chip and ranging applications (https://www.qorvo.com/products/p/DW1000#documents).
* The Qorvo forum (https://forum.qorvo.com/tags/c/wireless-connectivity/ultra-wideband/5/dw1000) and in particular the user AndyA (https://forum.qorvo.com/u/andya/summary) made this project possible.
* Information on the LoRa module can be found on the Seeed Studio Wiki (https://wiki.seeedstudio.com/wio_e5_class/).
