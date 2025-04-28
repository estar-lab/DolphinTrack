/* USER CODE BEGIN Header */
/**
 **************w***************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include "lcd.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define UART_RX_BUF_SIZE 1024
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc;

SPI_HandleTypeDef hspi1;
SPI_HandleTypeDef hspi2;
DMA_HandleTypeDef hdma_spi2_tx;

TIM_HandleTypeDef htim3;

UART_HandleTypeDef huart1;
DMA_HandleTypeDef hdma_usart1_rx;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_SPI1_Init(void);
static void MX_ADC_Init(void);
static void MX_SPI2_Init(void);
static void MX_TIM3_Init(void);
/* USER CODE BEGIN PFP */

void send_at_msg(char *msg)
{
  HAL_UART_Transmit_IT(&huart1, (uint8_t *)msg, strlen(msg));
}

void (*send_at_msg_ptr)(char *msg);

void node_recv(uint32_t timeout_ms)
{
  send_at_msg("AT+TEST=RXLRPKT\r\n");
  HAL_Delay(timeout_ms);
}

void node_send()
{
  uint16_t count = 0;
  char data[32];
  char cmd[128];

  memset(data, 0, sizeof(data));
  sprintf(data, "%04X", count);
  sprintf(cmd, "AT+TEST=TXLRPKT,\"5345454544%s\"\r\n", data);

  send_at_msg(cmd);
  HAL_Delay(100);
}

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

USBD_HandleTypeDef hUSBDDevice;
uint8_t uart_rx_buf[UART_RX_BUF_SIZE];

bool joined = 0;

char joined_msg[] = "+JOIN: Network joined";

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  send_at_msg_ptr = &send_at_msg;

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_USART1_UART_Init();
  MX_SPI1_Init();
  MX_ADC_Init();
  MX_SPI2_Init();
  MX_USB_DEVICE_Init();
  MX_TIM3_Init();
  /* USER CODE BEGIN 2 */

  // Power to screen
  HAL_GPIO_WritePin(SCREEN_EN_GPIO_Port, SCREEN_EN_Pin, GPIO_PIN_SET);
  HAL_GPIO_WritePin(SCREEN_EN_AUX_GPIO_Port, SCREEN_EN_AUX_Pin, GPIO_PIN_SET);

  LCD_init();

  // Startup sequence for LoRa module
  HAL_GPIO_WritePin(E5_NRST_GPIO_Port, E5_NRST_Pin, GPIO_PIN_SET);
  HAL_Delay(2000);
  HAL_GPIO_WritePin(E5_NRST_GPIO_Port, E5_NRST_Pin, GPIO_PIN_RESET);
  HAL_Delay(100);
  HAL_GPIO_WritePin(E5_NRST_GPIO_Port, E5_NRST_Pin, GPIO_PIN_SET);
  HAL_Delay(100);

  // LoRa command reception callback
  HAL_UARTEx_ReceiveToIdle_DMA(&huart1, uart_rx_buf, UART_RX_BUF_SIZE);

  // Startup lcd text

  UG_FontSetTransparency(1);
  UG_FillScreen(C_BLACK);

  LCD_PutStr(50, 20, "NEAR ANCHOR", FONT_16X26, C_WHITE, C_BLACK);
  LCD_PutStr(50, 50, "DESIGNED BY:", FONT_16X26, C_WHITE, C_BLACK);
  LCD_PutStr(50, 80, "TERRY TAO", FONT_16X26, C_WHITE, C_BLACK);
  LCD_PutStr(50, 110, "THOR HELGESON", FONT_16X26, C_WHITE, C_BLACK);

  //  char tx_msg4[] = "AT+DR=US915\r\n";
  //  HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg4, sizeof(tx_msg4));
  //  HAL_Delay(100);
  //
  //  char tx_msg5[] = "AT+CH\r\n";
  //  HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg5, sizeof(tx_msg5));
  //  HAL_Delay(100);
  //
  //  for(int i = 0; i < 72; i++)
  //  {
  //    if(i < 8 || i > 72)
  //    {
  //      char tx_msg5_[14];
  //      sprintf(tx_msg5_, "AT+CH=%d, OFF", i);
  //      HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg5_, sizeof(tx_msg5_));
  //    }
  //    HAL_Delay(50);
  //  }
  //
  //  char tx_msg0[] = "AT+KEY=APPKEY,\"B8AC2B18AE0F96A0FF83A63E33D0BA15\"\r\n";
  //  HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg0, sizeof(tx_msg0));
  //  HAL_Delay(500);
  //
  //  char tx_msg3[] = "AT+MODE= LWOTAA\r\n";
  //  HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg3, sizeof(tx_msg3));
  //  HAL_Delay(100);

  //  while(!joined)
  //  {
  //    char tx_msg_[] = "AT+ID=DevEui\r\n";
  //    HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg_, sizeof(tx_msg_));
  //    HAL_Delay(1000);
  //
  //    char tx_msg2[] = "AT+ID=AppEui\r\n";
  //    HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg2, sizeof(tx_msg2));
  //    HAL_Delay(1000);
  //
  //    char tx_msg_class[] = "AT+CLASS=C\r\n";
  //    HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg_class, sizeof(tx_msg_class));
  //    HAL_Delay(1000);

  // char tx_msg_mc[] = "AT+LW=MC,OFF\r\n";
  // HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg_class, sizeof(tx_msg_class));
  // HAL_Delay(1000);

  //    char tx_msg_durmx[] = "AT+LW=DUMRX,ON\r\n";
  //    HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg_durmx, sizeof(tx_msg_durmx));
  //    HAL_Delay(1000);
  //
  //    char tx_msg_dcmrx[] = "AT+LW=DCMRX,ON\r\n";
  //    HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg_dcmrx, sizeof(tx_msg_dcmrx));
  //    HAL_Delay(1000);
  //
  //    char tx_msg_dr[] = "AT+LW=CDR\r\n";
  //	HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg_dr, sizeof(tx_msg_dr));
  //	HAL_Delay(1000);
  //
  //	char tx_msg_adr[] = "AT+ADR=?\r\n";
  //	HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg_adr, sizeof(tx_msg_adr));
  //	HAL_Delay(1000);
  //
  //    char tx_msg8[] = "AT+JOIN\r\n";
  //    HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg8, sizeof(tx_msg8));
  //
  //    CDC_Transmit_FS("JOIN ATTEMPT\r\n", 14);
  //
  //    HAL_Delay(2000);
  //
  //  }

  //  CDC_Transmit_FS("JOIN SUCCESS\r\n", 14);

#ifdef FLASH_TAG
  tag_main();
#endif

#ifdef FLASH_ANCHOR
  anchor_main(send_at_msg_ptr);
#endif

#ifdef EX_02A_DEF
  dw_main();
#endif

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */

  while (1)
  {
    //	  send_at_msg("AT+TEST=TXLRSTR, \"Hello Cruel World\"\r\n");
    //	  HAL_Delay(100);

    //	  send_at_msg("AT+MODE=TEST\r\n");
    //	  HAL_Delay(100);

    //	  node_recv(2000);
    //	  node_send();

    //	   char tx_msg8[] = "AT\r\n";
    //	   HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg8, sizeof(tx_msg8));

    //     char tx_msg8[] = "AT+JOIN\r\n";
    //     HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg8, sizeof(tx_msg8));
    //     HAL_Delay(3000);

    //    char tx_msg[] = "AT+MSG=\"Adi<3robot\"\r\n";
    //    HAL_UART_Transmit_IT(&huart1, (uint8_t*) &tx_msg, sizeof(tx_msg));
    //    HAL_Delay(500);

    //	  send_at_msg("AT+MODE=TEST\r\n");
    //	  send_at_msg("AT+TEST=RFCFG,866,SF12,125,12,15,14,ON,OFF,OFF\r\n");

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI|RCC_OSCILLATORTYPE_HSI14
                              |RCC_OSCILLATORTYPE_HSI48;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSI48State = RCC_HSI48_ON;
  RCC_OscInitStruct.HSI14State = RCC_HSI14_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.HSI14CalibrationValue = 16;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  RCC_OscInitStruct.PLL.PREDIV = RCC_PREDIV_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USB|RCC_PERIPHCLK_USART1;
  PeriphClkInit.Usart1ClockSelection = RCC_USART1CLKSOURCE_PCLK1;
  PeriphClkInit.UsbClockSelection = RCC_USBCLKSOURCE_HSI48;

  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC_Init(void)
{

  /* USER CODE BEGIN ADC_Init 0 */

  /* USER CODE END ADC_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC_Init 1 */

  /* USER CODE END ADC_Init 1 */

  /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion)
  */
  hadc.Instance = ADC1;
  hadc.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV1;
  hadc.Init.Resolution = ADC_RESOLUTION_12B;
  hadc.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc.Init.ScanConvMode = ADC_SCAN_DIRECTION_FORWARD;
  hadc.Init.EOCSelection = ADC_EOC_SEQ_CONV;
  hadc.Init.LowPowerAutoWait = DISABLE;
  hadc.Init.LowPowerAutoPowerOff = DISABLE;
  hadc.Init.ContinuousConvMode = DISABLE;
  hadc.Init.DiscontinuousConvMode = DISABLE;
  hadc.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc.Init.DMAContinuousRequests = DISABLE;
  hadc.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  if (HAL_ADC_Init(&hadc) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel to be converted.
  */
  sConfig.Channel = ADC_CHANNEL_8;
  sConfig.Rank = ADC_RANK_CHANNEL_NUMBER;
  sConfig.SamplingTime = ADC_SAMPLETIME_1CYCLE_5;
  if (HAL_ADC_ConfigChannel(&hadc, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC_Init 2 */

  /* USER CODE END ADC_Init 2 */

}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{

  /* USER CODE BEGIN SPI1_Init 0 */

  /* USER CODE END SPI1_Init 0 */

  /* USER CODE BEGIN SPI1_Init 1 */

  /* USER CODE END SPI1_Init 1 */
  /* SPI1 parameter configuration*/
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_2;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 7;
  hspi1.Init.CRCLength = SPI_CRC_LENGTH_DATASIZE;
  hspi1.Init.NSSPMode = SPI_NSS_PULSE_ENABLE;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */

  /* USER CODE END SPI1_Init 2 */

}

/**
  * @brief SPI2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI2_Init(void)
{

  /* USER CODE BEGIN SPI2_Init 0 */

  /* USER CODE END SPI2_Init 0 */

  /* USER CODE BEGIN SPI2_Init 1 */

  /* USER CODE END SPI2_Init 1 */
  /* SPI2 parameter configuration*/
  hspi2.Instance = SPI2;
  hspi2.Init.Mode = SPI_MODE_MASTER;
  hspi2.Init.Direction = SPI_DIRECTION_2LINES;
  hspi2.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi2.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi2.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi2.Init.NSS = SPI_NSS_SOFT;
  hspi2.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_2;
  hspi2.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi2.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi2.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi2.Init.CRCPolynomial = 7;
  hspi2.Init.CRCLength = SPI_CRC_LENGTH_DATASIZE;
  hspi2.Init.NSSPMode = SPI_NSS_PULSE_ENABLE;
  if (HAL_SPI_Init(&hspi2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI2_Init 2 */

  /* USER CODE END SPI2_Init 2 */

}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 500-1;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 360-1;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 9600;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_RXOVERRUNDISABLE_INIT;
  huart1.AdvancedInit.OverrunDisable = UART_ADVFEATURE_OVERRUN_DISABLE;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Channel2_3_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel2_3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel2_3_IRQn);
  /* DMA1_Channel4_5_6_7_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel4_5_6_7_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel4_5_6_7_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, E5_NRST_Pin|DW_NSS_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, SCREEN_EN_AUX_Pin|SCREEN_EN_Pin|SD_CS_Pin|WAKE_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, TFT_CS_Pin|TFT_DC_Pin|TFT_RST_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin : E5_NRST_Pin */
  GPIO_InitStruct.Pin = E5_NRST_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(E5_NRST_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : DW_RESET_Pin */
  GPIO_InitStruct.Pin = DW_RESET_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(DW_RESET_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : DW_NSS_Pin */
  GPIO_InitStruct.Pin = DW_NSS_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
  HAL_GPIO_Init(DW_NSS_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : SCREEN_EN_AUX_Pin SCREEN_EN_Pin WAKE_Pin */
  GPIO_InitStruct.Pin = SCREEN_EN_AUX_Pin|SCREEN_EN_Pin|WAKE_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : SD_CS_Pin TFT_DC_Pin */
  GPIO_InitStruct.Pin = SD_CS_Pin|TFT_DC_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : TFT_CS_Pin TFT_RST_Pin */
  GPIO_InitStruct.Pin = TFT_CS_Pin|TFT_RST_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : BTN_DISP_Pin */
  GPIO_InitStruct.Pin = BTN_DISP_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(BTN_DISP_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : DW_IRQn_Pin */
  GPIO_InitStruct.Pin = DW_IRQn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(DW_IRQn_GPIO_Port, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI2_3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI2_3_IRQn);

  HAL_NVIC_SetPriority(EXTI4_15_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI4_15_IRQn);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t offset)
{

  static uint16_t last_offset = 0;

  // Ignore if called twice (which will happen on every half buffer)
  if (offset != last_offset)
  {

    // If wrap around reset last_size
    if (offset < last_offset)
      last_offset = 0;

    CDC_Transmit_FS(uart_rx_buf + last_offset, offset - last_offset);

    static uint16_t response_offset = 0;

    for (uint16_t i = last_offset; i < offset; i++)
    {
      if (uart_rx_buf[i] == '+')
      {
        response_offset = i;
        if (offset < response_offset)
          response_offset = 0;
        if (memcmp(uart_rx_buf + response_offset, joined_msg, sizeof(joined_msg) - 1) == 0)
        {
          joined = 1;
          break;
        }
      }
    }

    last_offset = offset;
  }
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
