#include "anchor.h"

#ifdef FLASH_ANCHOR

#include "id.h"
#include "lcd.h"

/* Example application name and version to display on LCD screen. */
#define APP_NAME "DS TWR RESP v1.2"

#define SCREEN_SLEEP_TIMEOUT 10000
#define TAG_ACTIVITY_TIMEOUT 10000

#define SLIDING_WINDOW_VARIANCE_RHO 0.15
#define SLIDING_WINDOW_VARIANCE_RHO_INV (1 - SLIDING_WINDOW_VARIANCE_RHO)

extern ADC_HandleTypeDef hadc;

uint32_t btn_press_tick = 0;

bool last_lcd_on = false;
bool last_btn = false;
bool disp_on = false;

uint8_t current_screen;
uint8_t total_screens;

uint32_t last_disp_tag_tick = 0;

void display_info()
{
    UG_FillFrame(146, 20, 210, 56, C_BLACK);
    uint16_t raw_volt_adc;
    char volt_msg[20];

    HAL_ADC_Start(&hadc);
    HAL_ADC_PollForConversion(&hadc, HAL_MAX_DELAY);
    raw_volt_adc = HAL_ADC_GetValue(&hadc);

    // Convert ADC value to battery percentage (example calculation)
    float battery_voltage = ((float)raw_volt_adc / 4095.0f) * 3.3f / 0.66f;

    // Format the battery percentage into a string
    sprintf(volt_msg, "BATT: %.2fV", battery_voltage);

    // Display the battery percentage on the LCD
    LCD_PutStr(50, 20, volt_msg, FONT_16X26, C_WHITE, C_BLACK);

    // Print the anchor ID (1-indexed)
    char anchor_id_msg[20];
    sprintf(anchor_id_msg, "ANCHOR ID: %d", ANCHOR_IDX + 1);
    LCD_PutStr(50, 70, anchor_id_msg, FONT_16X26, C_WHITE, C_BLACK);

    // Count the number of active tags
    uint8_t active_tags = 0;
    for (uint8_t i = 0; i < total_tags; i++)
    {
        if (tags_last_heard[i] != 0 && HAL_GetTick() - tags_last_heard[i] <= TAG_ACTIVITY_TIMEOUT)
        {
            active_tags++;
        }
    }

    UG_FillFrame(256, 120, 288, 146, C_BLACK);

    // Print the number of active tags
    char active_tags_msg[20];
    sprintf(active_tags_msg, "ACTIVE TAGS: %d", active_tags);
    LCD_PutStr(50, 120, active_tags_msg, FONT_16X26, C_WHITE, C_BLACK);
}

void display_tag(uint8_t tag_id)
{

    if (HAL_GetTick() - last_disp_tag_tick >= 100)
    {
        if (HAL_GetTick() - tags_last_heard[tag_id - 1] > TAG_ACTIVITY_TIMEOUT)
        {
            UG_FillFrame(152, 50, 300, 92, C_BLACK);
            char inactive_msg[20];
            sprintf(inactive_msg, "TAG %d: INACTIVE", tag_id);
            LCD_PutStr(50, 50, inactive_msg, FONT_16X26, C_WHITE, C_BLACK);
            last_disp_tag_tick = HAL_GetTick();
            return;
        }
        UG_FillFrame(152, 50, 300, 92, C_BLACK);
        char tag_msg[20];
        sprintf(tag_msg, "TAG %d: %.3f m", tag_id, tag_distances[tag_id - 1]);
        LCD_PutStr(50, 50, tag_msg, FONT_16X26, C_WHITE, C_BLACK);

        // Calculate the standard deviation (square root of variance)
        double tag_stdev = sqrt(tags_variance[tag_id - 1]);

        // Print the standard deviation of the tag
        UG_FillFrame(152, 100, 300, 142, C_BLACK);
        char stdev_msg[20];
        sprintf(stdev_msg, "STDEV: %.3f m", tag_stdev);
        LCD_PutStr(50, 100, stdev_msg, FONT_16X26, C_WHITE, C_BLACK);

        last_disp_tag_tick = HAL_GetTick();
    }
}

void display_clear()
{
    UG_FillScreen(C_BLACK);
}

void handle_screens()
{
    if (current_screen == 0)
    {
        display_info();
        return;
    }
    display_tag(current_screen);
}

void lcd_off()
{
    HAL_GPIO_WritePin(SCREEN_EN_GPIO_Port, SCREEN_EN_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(SCREEN_EN_AUX_GPIO_Port, SCREEN_EN_AUX_Pin, GPIO_PIN_RESET);
}

void lcd_on()
{
    HAL_GPIO_WritePin(SCREEN_EN_GPIO_Port, SCREEN_EN_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(SCREEN_EN_AUX_GPIO_Port, SCREEN_EN_AUX_Pin, GPIO_PIN_SET);

    LCD_init();
}

// Returns true when lcd should be on
bool handle_btn()
{
    if (HAL_GPIO_ReadPin(BTN_DISP_GPIO_Port, BTN_DISP_Pin))
    {
        btn_press_tick = HAL_GetTick();

        // Rising edge btn
        if (!last_btn && disp_on)
        {
            current_screen = (current_screen + 1) % total_screens;
            display_clear();
        }

        last_btn = true;
    }
    else
    {
        last_btn = false;
    }

    disp_on = HAL_GetTick() - btn_press_tick < SCREEN_SLEEP_TIMEOUT;

    if (disp_on)
    {
        // Rising edge lcd
        if (!last_lcd_on)
        {
            lcd_on();
            display_clear();
        }

        last_lcd_on = true;
        return true;
    }

    // Falling edge
    if (last_lcd_on)
    {
        lcd_off();
    }

    last_lcd_on = false;
    return false;
}

/* Default communication configuration. We use here EVK1000's default mode (mode 3). */
static dwt_config_t config = {
    4,               /* Channel number. */
    DWT_PRF_64M,     /* Pulse repetition frequency. */
    DWT_PLEN_1024,   /* Preamble length. Used in TX only. */
    DWT_PAC32,       /* Preamble acquisition chunk size. Used in RX only. */
    9,               /* TX preamble code. Used in TX only. */
    9,               /* RX preamble code. Used in RX only. */
    1,               /* 0 to use standard SFD, 1 to use non-standard SFD. */
    DWT_BR_850K,     /* Data rate. */
    DWT_PHRMODE_STD, /* PHY header mode. */
    (1025 + 64 - 32) /* SFD timeout (preamble length + 1 + SFD length - PAC size). Used in RX only. */
};

static dwt_txconfig_t tx_config = {
    TC_PGDELAY_CH4,
    0x00000000 /* Crank that shit */
};

/* Default antenna delay values for 64 MHz PRF. See NOTE 1 below. */
#define TX_ANT_DLY ANT_DLY
#define RX_ANT_DLY ANT_DLY

uint8 *anchor_id = anchor_addresses + ANCHOR_IDX * 2;

/* Frames used in the ranging process. See NOTE 2 below. */
static uint8 rx_poll_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 0, 0, 0, 0, 0x21, 0, 0};
static uint8 tx_resp_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 0, 0, 0, 0, 0x10, 0x02, 0, 0, 0, 0};
static uint8 rx_final_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 0, 0, 0, 0, 0x23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#define RX_POLL_MSG_TAG_ID_IDX 7
#define TX_RESP_MSG_TAG_ID_IDX 5
#define RX_FINAL_MSG_TAG_ID_IDX 7

#define RX_POLL_MSG_ANCHOR_ID_IDX 5
#define TX_RESP_MSG_ANCHOR_ID_IDX 7
#define RX_FINAL_MSG_ANCHOR_ID_IDX 5

/* Length of the common part of the message (up to and including the function code, see NOTE 2 below). */
#define ALL_MSG_COMMON_LEN 10
/* Index to access some of the fields in the frames involved in the process. */
#define ALL_MSG_SN_IDX 2
#define FINAL_MSG_POLL_TX_TS_IDX 10
#define FINAL_MSG_RESP_RX_TS_IDX 14
#define FINAL_MSG_FINAL_TX_TS_IDX 18
#define FINAL_MSG_TS_LEN 4
/* Frame sequence number, incremented after each transmission. */
static uint8 frame_seq_nb = 0;

/* Buffer to store received messages.
 * Its size is adjusted to longest frame that this example code is supposed to handle. */
#define RX_BUF_LEN 24
static uint8 rx_buffer[RX_BUF_LEN];

/* Hold copy of status register state here for reference so that it can be examined at a debug breakpoint. */
static uint32 status_reg = 0;

/* UWB microsecond (uus) to device time unit (dtu, around 15.65 ps) conversion factor.
 * 1 uus = 512 / 499.2 �s and 1 �s = 499.2 * 128 dtu. */
#define UUS_TO_DWT_TIME 65536

/* Delay between frames, in UWB microseconds. See NOTE 4 below. */
/* This is the delay from Frame RX timestamp to TX reply timestamp used for calculating/setting the DW1000's delayed TX function. This includes the
 * frame length of approximately 2.46 ms with above configuration. */
#define POLL_RX_TO_RESP_TX_DLY_UUS 2500
/* This is the delay from the end of the frame transmission to the enable of the receiver, as programmed for the DW1000's wait for response feature. */
#define RESP_TX_TO_FINAL_RX_DLY_UUS 100
/* Receive final timeout. See NOTE 5 below. */
#define FINAL_RX_TIMEOUT_UUS 2800
/* Preamble timeout, in multiple of PAC size. See NOTE 6 below. */
#define PRE_TIMEOUT 8

/* Timestamps of frames transmission/reception.
 * As they are 40-bit wide, we need to define a 64-bit int type to handle them. */
typedef signed long long int64;
typedef unsigned long long uint64;
static uint64 poll_rx_ts;
static uint64 resp_tx_ts;
static uint64 final_rx_ts;

/* Speed of light in air, in metres per second. */
#define SPEED_OF_LIGHT 299702547

/* Hold copies of computed time of flight and distance here for reference so that it can be examined at a debug breakpoint. */
static double tof;
static double distance;

/* String used to display measured distance on LCD screen (16 characters maximum). */
char dist_str[128] = {0};

/* Declaration of static functions. */
static uint64 get_tx_timestamp_u64(void);
static uint64 get_rx_timestamp_u64(void);
static void final_msg_get_ts(const uint8 *ts_field, uint32 *ts);

/*! ------------------------------------------------------------------------------------------------------------------
 * @fn main()
 *
 * @brief Application entry point.
 *
 * @param  none

 * @return none
 */
int anchor_main(void (*send_at_msg_ptr)(char *))
{
    total_screens = total_tags + 1;

    memcpy((rx_poll_msg) + RX_POLL_MSG_ANCHOR_ID_IDX, anchor_id, 2);
    memcpy((tx_resp_msg) + TX_RESP_MSG_ANCHOR_ID_IDX, anchor_id, 2);
    memcpy((rx_final_msg) + RX_FINAL_MSG_ANCHOR_ID_IDX, anchor_id, 2);
    /* Display application name on LCD. */
    // lcd_display_str(APP_NAME);

    /* Reset and initialise DW1000.
     * For initialisation, DW1000 clocks must be temporarily set to crystal speed. After initialisation SPI rate can be increased for optimum
     * performance. */
    reset_DW1000(); /* Target specific drive of RSTn line into DW1000 low for a period. */
    port_set_dw1000_slowrate();
    if (dwt_initialise(DWT_LOADUCODE) == DWT_ERROR)
    {
        // lcd_display_str("INIT FAILED");
        while (1)
        {
        };
    }
    port_set_dw1000_fastrate();

    /* Configure DW1000. See NOTE 7 below. */
    dwt_configure(&config);

    /* Apply TX config */
    dwt_configuretxrf(&tx_config);

    dwt_setdblrxbuffmode(0);

    /* Apply default antenna delay value. See NOTE 1 below. */
    dwt_setrxantennadelay(RX_ANT_DLY);
    dwt_settxantennadelay(TX_ANT_DLY);

    /* Set preamble timeout for expected frames. See NOTE 6 below. */
    // dwt_setpreambledetecttimeout(PRE_TIMEOUT);

    /* Set up LoRa transmitter. */
    HAL_Delay(2000);
    (*send_at_msg_ptr)("AT+MODE=TEST\r\n");
    HAL_Delay(100);
    (*send_at_msg_ptr)("AT+TEST=RFCFG,915,SF8,500,12,15,22,ON,OFF,OFF\r\n");

    /* Loop forever responding to ranging requests. */
    while (1)
    {

        /* Clear reception timeout to start next ranging process. */
        dwt_setrxtimeout(0);

        /* Activate reception immediately. */
        dwt_rxenable(DWT_START_RX_IMMEDIATE);

        /* Poll for reception of a frame or error/timeout. See NOTE 8 below. */
        while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) & (SYS_STATUS_RXFCG | SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR)))
        // while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) & (SYS_STATUS_RXFCG | SYS_STATUS_ALL_RX_ERR)))
        {

            if (handle_btn())
            {
                handle_screens();
            }
        };

        if (status_reg & SYS_STATUS_RXFCG)
        {
            uint32 frame_len;

            /* Clear good RX frame event in the DW1000 status register. */
            dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG);

            /* A frame has been received, read it into the local buffer. */
            frame_len = dwt_read32bitreg(RX_FINFO_ID) & RX_FINFO_RXFL_MASK_1023;
            if (frame_len <= RX_BUFFER_LEN)
            {
                dwt_readrxdata(rx_buffer, frame_len, 0);
            }

            /* Check that the frame is a poll sent by "DS TWR initiator" example.
             * As the sequence number field of the frame is not relevant, it is cleared to simplify the validation of the frame. */
            rx_buffer[ALL_MSG_SN_IDX] = 0;

            memcpy(tx_resp_msg + TX_RESP_MSG_TAG_ID_IDX, rx_buffer + RX_POLL_MSG_TAG_ID_IDX, 2);   // Save the incoming tag id into next message
            memcpy(rx_final_msg + RX_FINAL_MSG_TAG_ID_IDX, rx_buffer + RX_POLL_MSG_TAG_ID_IDX, 2); // Used for comparing tag id of poll with final
            memcpy(rx_poll_msg + RX_POLL_MSG_TAG_ID_IDX, rx_buffer + RX_POLL_MSG_TAG_ID_IDX, 2);   // Respond to all tags

            if (memcmp(rx_buffer, rx_poll_msg, ALL_MSG_COMMON_LEN) == 0)
            {

                uint32 resp_tx_time;
                int ret;

                /* Retrieve poll reception timestamp. */
                poll_rx_ts = get_rx_timestamp_u64();

                /* Set send time for response. See NOTE 9 below. */
                resp_tx_time = (poll_rx_ts + (POLL_RX_TO_RESP_TX_DLY_UUS * UUS_TO_DWT_TIME)) >> 8;
                dwt_setdelayedtrxtime(resp_tx_time);

                /* Set expected delay and timeout for final message reception. See NOTE 4 and 5 below. */
                dwt_setrxaftertxdelay(RESP_TX_TO_FINAL_RX_DLY_UUS);
                dwt_setrxtimeout(FINAL_RX_TIMEOUT_UUS);

                /* Write and send the response message. See NOTE 10 below.*/
                tx_resp_msg[ALL_MSG_SN_IDX] = frame_seq_nb;
                dwt_writetxdata(sizeof(tx_resp_msg), tx_resp_msg, 0); /* Zero offset in TX buffer. */
                dwt_writetxfctrl(sizeof(tx_resp_msg), 0, 1);          /* Zero offset in TX buffer, ranging. */
                ret = dwt_starttx(DWT_START_TX_DELAYED | DWT_RESPONSE_EXPECTED);

                /* If dwt_starttx() returns an error, abandon this ranging exchange and proceed to the next one. See NOTE 11 below. */
                if (ret == DWT_ERROR)
                {
                    continue;
                }

                /* Poll for reception of expected "final" frame or error/timeout. See NOTE 8 below. */
                while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) & (SYS_STATUS_RXFCG | SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR)))
                {
                };

                /* Increment frame sequence number after transmission of the response message (modulo 256). */
                frame_seq_nb++;

                if (status_reg & SYS_STATUS_RXFCG)
                {
                    int i;

                    for (i = 0; i < RX_BUF_LEN; i++)
                    {
                        rx_buffer[i] = 0;
                    }

                    /* Clear good RX frame event and TX frame sent in the DW1000 status register. */
                    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG | SYS_STATUS_TXFRS);

                    /* A frame has been received, read it into the local buffer. */
                    frame_len = dwt_read32bitreg(RX_FINFO_ID) & RX_FINFO_RXFLEN_MASK;
                    if (frame_len <= RX_BUF_LEN)
                    {
                        dwt_readrxdata(rx_buffer, frame_len, 0);
                    }


                    /* Check that the frame is a final message sent by "DS TWR initiator" example.
                     * As the sequence number field of the frame is not used in this example, it can be zeroed to ease the validation of the frame. */
                    rx_buffer[ALL_MSG_SN_IDX] = 0;

                    if (memcmp(rx_buffer, rx_final_msg, ALL_MSG_COMMON_LEN) == 0)
                    {
                        uint32 poll_tx_ts, resp_rx_ts, final_tx_ts;
                        uint32 poll_rx_ts_32, resp_tx_ts_32, final_rx_ts_32;
                        double Ra, Rb, Da, Db;
                        int64 tof_dtu;

                        /* Retrieve response transmission and final reception timestamps. */
                        resp_tx_ts = get_tx_timestamp_u64();
                        final_rx_ts = get_rx_timestamp_u64();

                        /* Get timestamps embedded in the final message. */
                        final_msg_get_ts(&rx_buffer[FINAL_MSG_POLL_TX_TS_IDX], &poll_tx_ts);
                        final_msg_get_ts(&rx_buffer[FINAL_MSG_RESP_RX_TS_IDX], &resp_rx_ts);
                        final_msg_get_ts(&rx_buffer[FINAL_MSG_FINAL_TX_TS_IDX], &final_tx_ts);

                        /* Compute time of flight. 32-bit subtractions give correct answers even if clock has wrapped. See NOTE 12 below. */
                        poll_rx_ts_32 = (uint32)poll_rx_ts;
                        resp_tx_ts_32 = (uint32)resp_tx_ts;
                        final_rx_ts_32 = (uint32)final_rx_ts;
                        Ra = (double)(resp_rx_ts - poll_tx_ts);
                        Rb = (double)(final_rx_ts_32 - resp_tx_ts_32);
                        Da = (double)(final_tx_ts - resp_rx_ts);
                        Db = (double)(resp_tx_ts_32 - poll_rx_ts_32);
                        tof_dtu = (int64)((Ra * Rb - Da * Db) / (Ra + Rb + Da + Db));

                        tof = tof_dtu * DWT_TIME_UNITS;

                        distance = tof * SPEED_OF_LIGHT;

                        if (distance < 0)
                        {
                            continue;
                        }

                        uint8 tag_index = rx_final_msg[RX_FINAL_MSG_TAG_ID_IDX + 1] - '0' - 1; // Convert char to uint8
                        if (tag_index < total_tags)                                            // Ensure tag ID is within bounds
                        {
                            tags_last_heard[tag_index] = HAL_GetTick();
                            tag_distances[tag_index] = distance; // Update the distance for the tag

                            // Update tag mean and variance
                            tags_variance[tag_index] = (SLIDING_WINDOW_VARIANCE_RHO_INV * tags_variance[tag_index]) + (SLIDING_WINDOW_VARIANCE_RHO * (distance - tags_mean[tag_index]) * (distance - tags_mean[tag_index]));
                            tags_mean[tag_index] = (SLIDING_WINDOW_VARIANCE_RHO_INV * tags_mean[tag_index]) + (SLIDING_WINDOW_VARIANCE_RHO * distance);
                        }

                        /* Create message for LoRa transmission */
                        // ID's and distances are repeated for naive error correction
                        sprintf(dist_str, "AT+TEST=TXLRSTR, \"%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f\"\r\n",
                        	anchor_id[1], rx_final_msg[RX_FINAL_MSG_TAG_ID_IDX + 1],
                            anchor_id[1], rx_final_msg[RX_FINAL_MSG_TAG_ID_IDX + 1],
                            anchor_id[1], rx_final_msg[RX_FINAL_MSG_TAG_ID_IDX + 1],
                            anchor_id[1], rx_final_msg[RX_FINAL_MSG_TAG_ID_IDX + 1],
                            anchor_id[1], rx_final_msg[RX_FINAL_MSG_TAG_ID_IDX + 1],
                            distance, distance, distance, distance, distance);

                        /* Transmit message over LoRa */
                        (*send_at_msg_ptr)(dist_str);
                    }
                }
                else
                {
                    /* Clear RX error/timeout events in the DW1000 status register. */
                    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);

                    /* Reset RX to properly reinitialise LDE operation. */
                    dwt_rxreset();
                }
            }
        }
        else
        {
            /* Clear RX error/timeout events in the DW1000 status register. */
            dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);

            /* Reset RX to properly reinitialise LDE operation. */
            dwt_rxreset();
        }
    }
}

/*! ------------------------------------------------------------------------------------------------------------------
 * @fn get_tx_timestamp_u64()
 *
 * @brief Get the TX time-stamp in a 64-bit variable.
 *        /!\ This function assumes that length of time-stamps is 40 bits, for both TX and RX!
 *
 * @param  none
 *
 * @return  64-bit value of the read time-stamp.
 */
static uint64 get_tx_timestamp_u64(void)
{
    uint8 ts_tab[5];
    uint64 ts = 0;
    int i;
    dwt_readtxtimestamp(ts_tab);
    for (i = 4; i >= 0; i--)
    {
        ts <<= 8;
        ts |= ts_tab[i];
    }
    return ts;
}

/*! ------------------------------------------------------------------------------------------------------------------
 * @fn get_rx_timestamp_u64()
 *
 * @brief Get the RX time-stamp in a 64-bit variable.
 *        /!\ This function assumes that length of time-stamps is 40 bits, for both TX and RX!
 *
 * @param  none
 *
 * @return  64-bit value of the read time-stamp.
 */
static uint64 get_rx_timestamp_u64(void)
{
    uint8 ts_tab[5];
    uint64 ts = 0;
    int i;
    dwt_readrxtimestamp(ts_tab);
    for (i = 4; i >= 0; i--)
    {
        ts <<= 8;
        ts |= ts_tab[i];
    }
    return ts;
}

/*! ------------------------------------------------------------------------------------------------------------------
 * @fn final_msg_get_ts()
 *
 * @brief Read a given timestamp value from the final message. In the timestamp fields of the final message, the least
 *        significant byte is at the lower address.
 *
 * @param  ts_field  pointer on the first byte of the timestamp field to read
 *         ts  timestamp value
 *
 * @return none
 */
static void final_msg_get_ts(const uint8 *ts_field, uint32 *ts)
{
    int i;
    *ts = 0;
    for (i = 0; i < FINAL_MSG_TS_LEN; i++)
    {
        *ts += ts_field[i] << (i * 8);
    }
}
#endif
/*****************************************************************************************************************************************************
 * NOTES:
 *
 * 1. The sum of the values is the TX to RX antenna delay, experimentally determined by a calibration process. Here we use a hard coded typical value
 *    but, in a real application, each device should have its own antenna delay properly calibrated to get the best possible precision when performing
 *    range measurements.
 * 2. The messages here are similar to those used in the DecaRanging ARM application (shipped with EVK1000 kit). They comply with the IEEE
 *    802.15.4 standard MAC data frame encoding and they are following the ISO/IEC:24730-62:2013 standard. The messages used are:
 *     - a poll message sent by the initiator to trigger the ranging exchange.
 *     - a response message sent by the responder allowing the initiator to go on with the process
 *     - a final message sent by the initiator to complete the exchange and provide all information needed by the responder to compute the
 *       time-of-flight (distance) estimate.
 *    The first 10 bytes of those frame are common and are composed of the following fields:
 *     - byte 0/1: frame control (0x8841 to indicate a data frame using 16-bit addressing).
 *     - byte 2: sequence number, incremented for each new frame.
 *     - byte 3/4: PAN ID (0xDECA).
 *     - byte 5/6: destination address, see NOTE 3 below.
 *     - byte 7/8: source address, see NOTE 3 below.
 *     - byte 9: function code (specific values to indicate which message it is in the ranging process).
 *    The remaining bytes are specific to each message as follows:
 *    Poll message:
 *     - no more data
 *    Response message:
 *     - byte 10: activity code (0x02 to tell the initiator to go on with the ranging exchange).
 *     - byte 11/12: activity parameter, not used for activity code 0x02.
 *    Final message:
 *     - byte 10 -> 13: poll message transmission timestamp.
 *     - byte 14 -> 17: response message reception timestamp.
 *     - byte 18 -> 21: final message transmission timestamp.
 *    All messages end with a 2-byte checksum automatically set by DW1000.
 * 3. Source and destination addresses are hard coded constants in this example to keep it simple but for a real product every device should have a
 *    unique ID. Here, 16-bit addressing is used to keep the messages as short as possible but, in an actual application, this should be done only
 *    after an exchange of specific messages used to define those short addresses for each device participating to the ranging exchange.
 * 4. Delays between frames have been chosen here to ensure proper synchronisation of transmission and reception of the frames between the initiator
 *    and the responder and to ensure a correct accuracy of the computed distance. The user is referred to DecaRanging ARM Source Code Guide for more
 *    details about the timings involved in the ranging process.
 * 5. This timeout is for complete reception of a frame, i.e. timeout duration must take into account the length of the expected frame. Here the value
 *    is arbitrary but chosen large enough to make sure that there is enough time to receive the complete final frame sent by the responder at the
 *    110k data rate used (around 3.5 ms).
 * 6. The preamble timeout allows the receiver to stop listening in situations where preamble is not starting (which might be because the responder is
 *    out of range or did not receive the message to respond to). This saves the power waste of listening for a message that is not coming. We
 *    recommend a minimum preamble timeout of 5 PACs for short range applications and a larger value (e.g. in the range of 50% to 80% of the preamble
 *    length) for more challenging longer range, NLOS or noisy environments.
 * 7. In a real application, for optimum performance within regulatory limits, it may be necessary to set TX pulse bandwidth and TX power, (using
 *    the dwt_configuretxrf API call) to per device calibrated values saved in the target system or the DW1000 OTP memory.
 * 8. We use polled mode of operation here to keep the example as simple as possible but all status events can be used to generate interrupts. Please
 *    refer to DW1000 User Manual for more details on "interrupts". It is also to be noted that STATUS register is 5 bytes long but, as the event we
 *    use are all in the first bytes of the register, we can use the simple dwt_read32bitreg() API call to access it instead of reading the whole 5
 *    bytes.
 * 9. Timestamps and delayed transmission time are both expressed in device time units so we just have to add the desired response delay to poll RX
 *    timestamp to get response transmission time. The delayed transmission time resolution is 512 device time units which means that the lower 9 bits
 *    of the obtained value must be zeroed. This also allows to encode the 40-bit value in a 32-bit words by shifting the all-zero lower 8 bits.
 * 10. dwt_writetxdata() takes the full size of the message as a parameter but only copies (size - 2) bytes as the check-sum at the end of the frame is
 *     automatically appended by the DW1000. This means that our variable could be two bytes shorter without losing any data (but the sizeof would not
 *     work anymore then as we would still have to indicate the full length of the frame to dwt_writetxdata()).
 * 11. When running this example on the EVB1000 platform with the POLL_RX_TO_RESP_TX_DLY response delay provided, the dwt_starttx() is always
 *     successful. However, in cases where the delay is too short (or something else interrupts the code flow), then the dwt_starttx() might be issued
 *     too late for the configured start time. The code below provides an example of how to handle this condition: In this case it abandons the
 *     ranging exchange and simply goes back to awaiting another poll message. If this error handling code was not here, a late dwt_starttx() would
 *     result in the code flow getting stuck waiting subsequent RX event that will will never come. The companion "initiator" example (ex_05a) should
 *     timeout from awaiting the "response" and proceed to send another poll in due course to initiate another ranging exchange.
 * 12. The high order byte of each 40-bit time-stamps is discarded here. This is acceptable as, on each device, those time-stamps are not separated by
 *     more than 2**32 device time units (which is around 67 ms) which means that the calculation of the round-trip delays can be handled by a 32-bit
 *     subtraction.
 * 13. The user is referred to DecaRanging ARM application (distributed with EVK1000 product) for additional practical example of usage, and to the
 *     DW1000 API Guide for more details on the DW1000 driver functions.
 ****************************************************************************************************************************************************/
