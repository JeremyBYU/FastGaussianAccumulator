#include "nanoflann.hpp"
#include "FastGA/Helper.hpp"

namespace FastGA
{

namespace NFA
{

template <typename Derived>
struct BucketAdaptor
{
    const Derived& obj;

    /// The constructor that sets the data set source
    BucketAdaptor(const Derived& obj_) : obj(obj_) {}

    /// CRTP helper method
    inline const Derived& derived() const { return obj; }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return derived().size(); }

    // Returns the dim'th component of the idx'th point in the class:
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return derived()[idx].normal[0];
        else if (dim == 1)
            return derived()[idx].normal[1];
        else
            return derived()[idx].normal[2];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const
    {
        return false;
    }

}; // end of BucketAdaptor

typedef BucketAdaptor<std::vector<FastGA::Bucket<uint32_t>>> BUCKET2KD;
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, BUCKET2KD>,
    BUCKET2KD,
    3 /* dim */
    >
    nano_kd_tree_t;

}
}
