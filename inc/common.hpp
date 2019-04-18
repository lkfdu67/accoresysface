//
// Created by hua on 19-4-15.
//

#ifndef LOADPARAM_COMMON_HPP
#define LOADPARAM_COMMON_HPP

//#include <gflags/gflags.h>
//#include <glog/logging.h>


// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>


#endif //LOADPARAM_COMMON_HPP