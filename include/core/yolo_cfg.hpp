#pragma once

#include "yolo.h"

void option_insert(list *l, char *key, char *val);
int read_option(char *s, list *options);
void option_unused(list *l);
char *option_find(list *l, const char *key);
char *option_find_str(list *l, const char *key, char *def);
int option_find_int(list *l, const char *key, int def);
int option_find_int_quiet(list *l, const char *key, int def);
float option_find_float_quiet(list *l, const char *key, float def);
float option_find_float(list *l, const char *key, float def);
list *read_data_cfg(char *filename);
