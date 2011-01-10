struct Slicing
{
  int nslices;
  int *slice_ptr, *slices;
};

#ifdef __cplusplus
extern "C"
#endif
struct Slicing *slice(int, int *, int *, int, int *,
		      int, int *);

#ifdef __cplusplus
extern "C"
#endif
void freeSlicing(struct Slicing *);
