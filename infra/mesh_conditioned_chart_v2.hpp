// Prepared source chart only. Sterbenz, Floating-Point Computation (1974), 4.3.1.
// Included inside conditioned_qem after the authenticated serialization core.
#ifndef WR_CONDITIONED_CHART_V2_SHA256
#error Bind the exact new chart header
#endif
#ifndef WR_CONDITIONED_CHART_V2_POLICY_SHA256
#error Bind the Python/native chart policy identity
#endif
#ifdef __FAST_MATH__
#error Exact source chart forbids fast-math
#endif

struct Chart {
  Eigen::RowVector3d origin;
  std::array<bool,3> midpoint_selected{};
  double scale=0;
  int exponent=0;
  uint64_t source_vertices=0;
  bool byte_exact=true;

  static serialization_qem::Integer double_units(double value) {
    uint64_t bits; std::memcpy(&bits,&value,sizeof(bits));
    const uint64_t e=(bits>>52)&2047, fraction=bits&0xfffffffffffffULL;
    if(e==2047) throw std::runtime_error("Nonfinite Sterbenz operand");
    serialization_qem::Integer result=e ? (fraction|0x10000000000000ULL) : fraction;
    if(e) result<<=(e-1); // Exact common unit 2^-1074; gradual underflow.
    return (bits>>63) ? -result : result;
  }

  static bool sterbenz(const Eigen::MatrixXd& V,int axis,double candidate) {
    if(candidate==0) return true;
    double smallest=std::numeric_limits<double>::infinity(),largest=0;
    for(int v=0;v<V.rows();++v) {
      const double x=V(v,axis);
      if(!std::isfinite(x)) throw std::runtime_error("Nonfinite source chart row");
      if((candidate>0 && !(x>0)) || (candidate<0 && !(x<0))) return false;
      const double a=std::abs(x); smallest=std::min(smallest,a); largest=std::max(largest,a);
    }
    const auto y=double_units(std::abs(candidate));
    return 2*double_units(smallest)>=y && double_units(largest)<=2*y;
  }

  Eigen::RowVector3d encode(const Eigen::RowVector3d& p) const {
    Eigen::RowVector3d result;
    for(int j=0;j<3;++j) {
      volatile double delta=origin[j]==0 ? p[j] : p[j]-origin[j];
      volatile double x=std::ldexp(double(delta),-exponent); result[j]=x;
      if(!std::isfinite(delta) || !std::isfinite(x) || (delta!=0 && x==0))
        throw std::runtime_error("Chart v2 encode overflow or nonzero underflow");
    }
    return result;
  }

  Eigen::RowVector3d decode(const Eigen::RowVector3d& p) const {
    Eigen::RowVector3d result;
    for(int j=0;j<3;++j) {
      volatile double delta=std::ldexp(p[j],exponent);
      volatile double x=origin[j]==0 ? double(delta) : origin[j]+delta; result[j]=x;
      if(!std::isfinite(delta) || !std::isfinite(x) || (p[j]!=0 && delta==0))
        throw std::runtime_error("Chart v2 decode overflow or nonzero underflow");
    }
    return result;
  }

  Eigen::MatrixXd decode_matrix(const Eigen::MatrixXd& V) const {
    Eigen::MatrixXd physical(V.rows(),3);
    for(int v=0;v<V.rows();++v) physical.row(v)=decode(Eigen::RowVector3d(V.row(v)));
    return physical;
  }

  Chart(const Eigen::MatrixXd& V,const Eigen::MatrixXi& F,Eigen::MatrixXd& canonical) {
    if(sizeof(double)!=8 || !std::numeric_limits<double>::is_iec559
        || std::numeric_limits<double>::digits!=53 || std::fegetround()!=FE_TONEAREST)
      throw std::runtime_error("Chart v2 requires IEEE binary64 and nearest rounding");
    volatile double tiny=std::numeric_limits<double>::denorm_min(),one=1.;
    volatile double kept=tiny*one;
    if(!(tiny>0) || kept!=tiny) throw std::runtime_error("Chart v2 requires gradual binary64 underflow");
    if(V.rows()<4 || V.cols()!=3 || !V.allFinite() || F.rows()==0 || F.cols()!=3
        || F.minCoeff()<0 || F.maxCoeff()>=V.rows()) throw std::runtime_error("Invalid chart source arrays");
    Eigen::RowVector3d lo=V.row(F(0,0)),hi=lo,midpoint;
    for(int f=0;f<F.rows();++f) for(int j=0;j<3;++j) {
      lo=lo.cwiseMin(V.row(F(f,j))); hi=hi.cwiseMax(V.row(F(f,j)));
    }
    double maximum=0;
    for(int j=0;j<3;++j) {
      volatile double extent=hi[j]-lo[j],half=extent*.5;
      volatile double middle=lo[j]+half; midpoint[j]=middle;
      if(!std::isfinite(extent) || !std::isfinite(middle))
        throw std::runtime_error("Chart v2 bbox outside finite binary64");
      maximum=std::max(maximum,double(extent));
    }
    if(!(maximum>0)) throw std::runtime_error("Chart v2 requires positive source extent");
    const double mantissa=std::frexp(maximum,&exponent);
    exponent-=int(mantissa==.5);
    if(exponent < -1074 || exponent > 1023) throw std::runtime_error("Chart v2 covering scale unrepresentable");
    scale=std::ldexp(1.,exponent);
    // The choice depends ONLY on sufficient source conditions, never inverse success.
    for(int j=0;j<3;++j) {
      midpoint_selected[j]=midpoint[j]!=0 && sterbenz(V,j,midpoint[j]);
      origin[j]=midpoint_selected[j] ? midpoint[j] : 0.;
    }
    canonical.resize(V.rows(),3); source_vertices=V.rows();
    for(int v=0;v<V.rows();++v) {
      const Eigen::RowVector3d p=V.row(v),x=encode(p),roundtrip=decode(x);
      for(int j=0;j<3;++j) {
        if(roundtrip[j]!=p[j]) throw std::runtime_error("Selected chart v2 inverse not exact; no alternate origin");
        byte_exact &= std::memcmp(&roundtrip[j],&p[j],sizeof(double))==0;
      }
      canonical.row(v)=x;
    }
  }

  void summary(std::ostream& out) const {
    out<<std::setprecision(17)<<"{\"chart_version\":2,\"header_sha256\":\""<<WR_CONDITIONED_CHART_V2_SHA256
       <<"\",\"policy_sha256\":\""<<WR_CONDITIONED_CHART_V2_POLICY_SHA256<<"\",\"origin\":["
       <<origin[0]<<','<<origin[1]<<','<<origin[2]<<"],\"origin_modes\":[";
    for(int j=0;j<3;++j) out<<(j ? "," : "")<<'"'<<(midpoint_selected[j] ? "sterbenz_midpoint" : "zero")<<'"';
    out<<"],\"scale\":"<<scale<<",\"scale_exponent\":"<<exponent
       <<",\"source_roundtrip_vertices\":"<<source_vertices
       <<",\"chart_scale_positive\":true,\"source_roundtrip_numerically_exact\":true,"
         "\"source_roundtrip_byte_exact\":"<<(byte_exact?"true":"false")
       <<",\"origin_search_performed\":false,\"physical_geometry_rescaled\":false,"
         "\"new_numeric_algorithm\":true,\"native_qslim_implementation_reused\":true,"
         "\"chart_refitted\":false,\"native_backend_qualified\":false,\"adopted\":false}";
  }
};
