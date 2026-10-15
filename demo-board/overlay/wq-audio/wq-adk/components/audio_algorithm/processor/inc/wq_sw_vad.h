#ifndef _WQ_SW_VAD_H
#define _WQ_SW_VAD_H
extern int wq_get_sw_vad_hd_size(void);
extern int wq_get_sw_vad_scratch_size(void);
extern void *wq_sw_vad_init(void *vad_hd, void *pscratch);
extern int wq_sw_vad_process(void *vad_hd, char *in_data, unsigned int in_len);
extern void wq_get_sw_vad_lib_version(char *version_string);
#endif /* _WQ_SW_VAD_H */
