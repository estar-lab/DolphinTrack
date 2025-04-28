/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f0xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include "port.h"

#define ANT_DLY 16433 // 16436

// #define FLASH_TAG
#define FLASH_ANCHOR

// # define ANCHOR_ID 0x0       // TWR with allocations, id from 0x0 - 0xF
// # define TAG_ID    0x0       // TWR with allocations, id from 0x0 - 0xF
//  #define EX_01A_DEF // TRANSMITTER SIMPLE
// #define EX_02A_DEF 1 // RECEIVER SIMPLE
// #define EX_02E_DEF 1 // RECEIVER DOUBLE BUFFER
// #define EX_05A_DEF 1 // TWR INITIALIZER
// #define EX_05B_DEF 1 // TWR RESPONDER
// # define EX_03A_DEF 1 // TX WAIT
// # define EX_03B_DEF 1 // RX WAIT
// #define RESPONDER_NUM 0
// #define TOTAL_RESPONDERS 1


/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define E5_NRST_Pin GPIO_PIN_0
#define E5_NRST_GPIO_Port GPIOA
#define DW_RESET_Pin GPIO_PIN_2
#define DW_RESET_GPIO_Port GPIOA
#define DW_RESET_EXTI_IRQn EXTI2_3_IRQn
#define DW_NSS_Pin GPIO_PIN_4
#define DW_NSS_GPIO_Port GPIOA
#define DW_SCK_Pin GPIO_PIN_5
#define DW_SCK_GPIO_Port GPIOA
#define DW_MISO_Pin GPIO_PIN_6
#define DW_MISO_GPIO_Port GPIOA
#define DW_MOSI_Pin GPIO_PIN_7
#define DW_MOSI_GPIO_Port GPIOA
#define VBATT_ADC_Pin GPIO_PIN_0
#define VBATT_ADC_GPIO_Port GPIOB
#define SCREEN_EN_AUX_Pin GPIO_PIN_1
#define SCREEN_EN_AUX_GPIO_Port GPIOB
#define SCREEN_EN_Pin GPIO_PIN_2
#define SCREEN_EN_GPIO_Port GPIOB
#define SD_CS_Pin GPIO_PIN_11
#define SD_CS_GPIO_Port GPIOB
#define TFT_CS_Pin GPIO_PIN_12
#define TFT_CS_GPIO_Port GPIOB
#define BTN_DISP_Pin GPIO_PIN_13
#define BTN_DISP_GPIO_Port GPIOB
#define DW_IRQn_Pin GPIO_PIN_5
#define DW_IRQn_GPIO_Port GPIOB
#define DW_IRQn_EXTI_IRQn EXTI4_15_IRQn
#define WAKE_Pin GPIO_PIN_6
#define WAKE_GPIO_Port GPIOB
#define TFT_DC_Pin GPIO_PIN_8
#define TFT_DC_GPIO_Port GPIOB
#define TFT_RST_Pin GPIO_PIN_9
#define TFT_RST_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
