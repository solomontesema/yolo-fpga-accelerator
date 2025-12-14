#include <core/yolo.h>

#include <cassert>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

void malloc_error()
{
    std::fprintf(stderr, "Malloc error\n");
    std::exit(-1);
}

void free_node(node *n)
{
    node *next;
    while (n) {
        next = n->next;
        free(n);
        n = next;
    }
}

} // namespace

void file_error(char *s)
{
    std::fprintf(stderr, "Couldn't open file: %s\n", s);
    std::exit(0);
}

void error(const char *s)
{
    perror(s);
    assert(0);
    std::exit(-1);
}

list *make_list()
{
    list *l = (list *)malloc(sizeof(list));
    l->size = 0;
    l->front = 0;
    l->back = 0;
    return l;
}

void *list_pop(list *l)
{
    if (!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if (l->back) l->back->next = 0;
    free(b);
    --l->size;

    return val;
}

void list_insert(list *l, void *val)
{
    node *new_node = (node *)malloc(sizeof(node));
    new_node->val = val;
    new_node->next = 0;

    if (!l->back) {
        l->front = new_node;
        new_node->prev = 0;
    } else {
        l->back->next = new_node;
        new_node->prev = l->back;
    }
    l->back = new_node;
    ++l->size;
}

void free_list(list *l)
{
    free_node(l->front);
    free(l);
}

void free_list_contents(list *l)
{
    node *n = l->front;
    while (n) {
        free(n->val);
        n = n->next;
    }
}

void **list_to_array(list *l)
{
    void **a = (void **)calloc(l->size, sizeof(void *));
    int count = 0;
    node *n = l->front;
    while (n) {
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}

void del_arg(int argc, char **argv, int index)
{
    int i;
    for (i = index; i < argc - 1; ++i) argv[i] = argv[i + 1];
    argv[i] = 0;
}

int find_arg(int argc, char *argv[], const char *arg)
{
    int i;
    for (i = 0; i < argc; ++i) {
        if (!argv[i]) continue;
        if (0 == strcmp(argv[i], arg)) {
            del_arg(argc, argv, i);
            return 1;
        }
    }
    return 0;
}

int find_int_arg(int argc, char **argv, const char *arg, int def)
{
    int i;
    for (i = 0; i < argc - 1; ++i) {
        if (!argv[i]) continue;
        if (0 == strcmp(argv[i], arg)) {
            def = atoi(argv[i + 1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

float find_float_arg(int argc, char **argv, const char *arg, float def)
{
    int i;
    for (i = 0; i < argc - 1; ++i) {
        if (!argv[i]) continue;
        if (0 == strcmp(argv[i], arg)) {
            def = atof(argv[i + 1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

char *find_char_arg(int argc, char **argv, const char *arg, char *def)
{
    int i;
    for (i = 0; i < argc - 1; ++i) {
        if (!argv[i]) continue;
        if (0 == strcmp(argv[i], arg)) {
            def = argv[i + 1];
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

unsigned char *read_file(char *filename)
{
    FILE *fp = fopen(filename, "rb");
    size_t size;

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char *text = (unsigned char *)calloc(size + 1, sizeof(unsigned char));
    size_t rsz = fread(text, 1, size, fp);
    (void)rsz;
    fclose(fp);
    return text;
}

list *split_str(char *s, char delim)
{
    size_t i;
    size_t len = strlen(s);
    list *l = make_list();
    list_insert(l, s);
    for (i = 0; i < len; ++i) {
        if (s[i] == delim) {
            s[i] = '\0';
            list_insert(l, &(s[i + 1]));
        }
    }
    return l;
}

void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for (i = 0; i < len; ++i) {
        char c = s[i];
        if (c == ' ' || c == '\t' || c == '\n') ++offset;
        else s[i - offset] = c;
    }
    s[len - offset] = '\0';
}

void strip_char(char *s, char bad)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for (i = 0; i < len; ++i) {
        char c = s[i];
        if (c == bad) ++offset;
        else s[i - offset] = c;
    }
    s[len - offset] = '\0';
}

void free_ptrs(void **ptrs, int n)
{
    int i;
    for (i = 0; i < n; ++i) free(ptrs[i]);
    free(ptrs);
}

char *fgetl(FILE *fp)
{
    if (feof(fp)) return 0;
    size_t size = 512;
    char *line = (char *)malloc(size * sizeof(char));
    if (!fgets(line, size, fp)) {
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while ((line[curr - 1] != '\n') && !feof(fp)) {
        if (curr == size - 1) {
            size *= 2;
            line = (char *)realloc(line, size * sizeof(char));
            if (!line) {
                printf("%ld\n", size);
                malloc_error();
            }
        }
        size_t readsize = size - curr;
        if (readsize > INT_MAX) readsize = INT_MAX - 1;
        char *rv = fgets(&line[curr], readsize, fp);
        (void)rv;
        curr = strlen(line);
    }
    if (line[curr - 1] == '\n') line[curr - 1] = '\0';

    return line;
}
