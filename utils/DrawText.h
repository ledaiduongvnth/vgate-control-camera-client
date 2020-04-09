//
// Created by d on 08/04/2020.
//

#ifndef CAMERA_CLIENT_DRAWTEXT_H
#define CAMERA_CLIENT_DRAWTEXT_H
#include "ft2build.h"
#include FT_FREETYPE_H

class DrawText {
public:
    FT_Library library;
    FT_Face face;
    int pixel_width;

    DrawText(int pixel_width){
        this->pixel_width = pixel_width;
        FT_Init_FreeType( &library );
        FT_New_Face( library,"../utils/Roboto.ttf",0,&face );
        FT_Set_Pixel_Sizes(face, this->pixel_width,0);
        FT_Select_Charmap(face, FT_ENCODING_UNICODE);
    }

    void my_draw_bitmap(Mat &img, FT_Bitmap *bitmap, int x, int y, Scalar color) {
        Scalar src_col, dst_col;
        for (int i = 0; i < bitmap->rows; i++) {
            for (int j = 0; j < bitmap->width; j++) {
                if (0 < i + y && i + y < img.cols){
                    unsigned char val = bitmap->buffer[j + i * bitmap->pitch];
                    float mix = (float) val / 255.0;
                    if (val != 0) {
                        src_col = Scalar(img.at<Vec3b>(i + y, j + x));
                        dst_col = mix * color + (1.0 - mix) * src_col;
                        img.at<Vec3b>(i + y, j + x) = Vec3b(dst_col[0], dst_col[1], dst_col[2]);
                    }
                }
            }
        }
    }

    float PrintString(Mat &img, std::wstring str, int x, int y, Scalar color) {
        FT_Bool use_kerning = 0;
        FT_UInt previous = 0;
        use_kerning = FT_HAS_KERNING(face);
        float prev_yadv = 0;
        float posx = 0;
        float posy = 0;
        float dx = 0;
        for (int k = 0; k < str.length(); k++) {
            int glyph_index = FT_Get_Char_Index(face, str.c_str()[k]);
            FT_GlyphSlot slot = face->glyph;  // a small shortcut
            if (k > 0) { dx = slot->advance.x / 64; }
            FT_Load_Glyph(face, glyph_index, FT_LOAD_DEFAULT);
            FT_Render_Glyph(slot, FT_RENDER_MODE_NORMAL);
            prev_yadv = slot->metrics.vertAdvance / 64;
            if (use_kerning && previous && glyph_index) {
                FT_Vector delta;
                FT_Get_Kerning(face, previous, glyph_index, FT_KERNING_DEFAULT, &delta);
                posx += (delta.x / 64);
            }
            posx += (dx);
            my_draw_bitmap(img, &slot->bitmap, posx + x + slot->bitmap_left, y - slot->bitmap_top + posy, color);
            previous = glyph_index;
        }
        return prev_yadv;
    }

    int GetWidth(std::wstring str){

    }

    void PrintText(Mat &img, std::wstring str, int x, int y, Scalar color) {
        float posy = 0;
        for (int pos = str.find_first_of(L'\n'); pos != wstring::npos; pos = str.find_first_of(L'\n')) {
            std::wstring substr = str.substr(0, pos);
            str.erase(0, pos + 1);
            posy += PrintString(img, substr, x, y + posy, color);
        }
        PrintString(img, str, x, y + posy, color);
    }
};


#endif //CAMERA_CLIENT_DRAWTEXT_H
