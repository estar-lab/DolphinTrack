#ifndef ID_H
#define ID_H

#include <stdint.h>

#define ANCHOR_IDX 2 // Which anchor are we flashing?
#define TAG_IDX 0    // Which tag are we flashing?

uint8_t anchor_addresses[] = {'0', '1',
                              '0', '2',
                              '0', '3',
							  '0', '4',
							  '0', '5'};

uint8_t tag_addresses[] = {'0', '1',
						   '0', '2',
						   '0', '3'};

double tag_distances [] = {0, 0, 0};
uint32_t tags_last_heard [] = {0, 0, 0};
double tags_mean [] = {0, 0, 0};
double tags_variance [] = {0, 0, 0};

uint8_t total_anchors = sizeof(anchor_addresses) / 2; // How many anchors in environment
uint8_t total_tags = sizeof(tag_addresses) / 2;       // How many tags in environment

uint8_t * get_anchor_id(uint8_t idx)
{
    return anchor_addresses + (idx * 2);
}

// uint8_t * get_tag_id(uint8_t idx)
// {
//     return tag_addresses + (idx * 2);
// }


// #ifdef FLASH_ANCHOR
//     uint8_t * anchor_id = get_anchor_id(ANCHOR_IDX); // ID of anchor we are flashing
// #endif

// #ifdef FLASH_TAG
//     uint8_t * tag_id = get_tag_id(TAG_IDX); // ID of tag we are flashing
// #endif

#endif
