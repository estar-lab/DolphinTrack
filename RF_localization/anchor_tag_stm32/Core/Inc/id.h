#ifndef ID_H
#define ID_H

#include <stdint.h>

#define ANCHOR_IDX 0 // Which anchor are we flashing?
#define TAG_IDX 0    // Which tag are we flashing?

/* List of all anchor ID's. The tag will attempt to range with each ID in this list. */
uint8_t anchor_addresses[] = {'0', '1',
                              '0', '2',
                              '0', '3',
							  '0', '4',
							  '0', '5'};

/* List of all tag addresses. This is not currently used to control the tag ranging order. */
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

#endif
