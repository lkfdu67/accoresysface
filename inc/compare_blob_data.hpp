//
// Created by jbk on 19-4-16.
//

#ifndef LOADPARAM_SAVE_DATA_HPP
#define LOADPARAM_SAVE_DATA_HPP



#include <utility>
#include <memory>
#include <string>
#include <common.hpp>
#include <blob_.hpp>

using asr::Blob;
using std::fstream;
using std::iostream;
using std::string;
using std::cout;
using std::endl;

namespace asr{
    template<typename DType>
    class CompareTopBlob{
    public:
        CompareTopBlob(){}
        void CompareInit(const string& load_folder,const vector<Blob<DType>* >& net_output_blobs, const vector<string>& layer_names);
        void SaveInit(const string& save_folder, const vector<Blob<DType>* >& net_output_blobs, const vector<string>& layer_names);
        ~CompareTopBlob(){}
        void SaveTopBlob();
        void LoadTopBlob(const string& load_folder);
        vector<DType> Compare();

        vector<DType> GetMaxError(){
            return this->max_error_;
        }
        void LoadShapes(const string& load_folder);

        void PrintVector(const vector<DType>& shape){
            for(int i = 0; i < shape.size(); ++i){
                cout << shape[i] << "\t";
            }
            cout << endl;
        }

    private:
        vector<string> layer_names_;
        vector<string> load_layer_names_;
        string save_folder_;
        vector<vector<int>> load_shapes_;
        vector<vector<int>> shapes_;
        vector<Blob<DType>* > net_output_blobs_;
        vector<Blob<DType>> load_output_blobs_;
        vector<Blob<DType>> error_blobs_;
        vector<DType> mse_;
        vector<DType> max_error_;
        vector<DType> mid_error_;
        vector<DType> min_error_;
    };
}


#endif //LOADPARAM_SAVE_DATA_HPP
