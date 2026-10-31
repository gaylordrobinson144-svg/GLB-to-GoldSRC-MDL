#define _CRT_SECURE_NO_WARNINGS

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace cg {
#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
}

#pragma pack(push, 1)
struct StudioHdr
{
    int   id, version;
    char  name[64];
    int   length;
    float eyeposition[3], min[3], max[3], bbmin[3], bbmax[3];
    int   flags;
    int   numbones, boneindex;
    int   numbonecontrollers, bonecontrollerindex;
    int   numhitboxes, hitboxindex;
    int   numseq, seqindex;
    int   numseqgroups, seqgroupindex;
    int   numtextures, textureindex, texturedataindex;
    int   numskinref, numskinfamilies, skinindex;
    int   numbodyparts, bodypartindex;
    int   numattachments, attachmentindex;
    int   soundtable, soundindex, soundgroups, soundgroupindex;
    int   numtransitions, transitionindex;
};
struct StudioBone
{
    char  name[32];
    int   parent, flags;
    int   bonecontroller[6];
    float value[6], scale[6];
};
struct StudioSeqDesc
{
    char  label[32];
    float fps;
    int   flags, activity, actweight;
    int   numevents, eventindex;
    int   numframes;
    int   numpivots, pivotindex;
    int   motiontype, motionbone;
    float linearmovement[3];
    int   automoveposindex, automoveangleindex;
    float bbmin[3], bbmax[3];
    int   numblends, animindex;
    int   blendtype[2];
    float blendstart[2], blendend[2];
    int   blendparent, seqgroup, entrynode, exitnode, nodeflags, nextseq;
};
struct StudioSeqGroup { char label[32]; char name[64]; int unused1, unused2; };
struct StudioTexture { char name[64]; int flags, width, height, index; };
struct StudioBodypart { char name[64]; int nummodels, base, modelindex; };
struct StudioModel
{
    char  name[64];
    int   type;
    float boundingradius;
    int   nummesh, meshindex;
    int   numverts, vertinfoindex, vertindex;
    int   numnorms, norminfoindex, normindex;
    int   numgroups, groupindex;
};
struct StudioMesh { int numtris, triindex, skinref, numnorms, normindex; };
struct StudioTriVert { short vertindex, normindex, s, t; };
#pragma pack(pop)

static_assert(sizeof(StudioHdr) == 244, "studiohdr_t");
static_assert(sizeof(StudioBone) == 112, "mstudiobone_t");
static_assert(sizeof(StudioSeqDesc) == 176, "mstudioseqdesc_t");
static_assert(sizeof(StudioSeqGroup) == 104, "mstudioseqgroup_t");
static_assert(sizeof(StudioTexture) == 80, "mstudiotexture_t");
static_assert(sizeof(StudioBodypart) == 76, "mstudiobodyparts_t");
static_assert(sizeof(StudioModel) == 112, "mstudiomodel_t");
static_assert(sizeof(StudioMesh) == 20, "mstudiomesh_t");
static_assert(sizeof(StudioTriVert) == 8, "mstudiotrivert_t");

static const int   STUDIO_NF_MASKED = 0x0040;
static const int   STUDIO_LOOPING = 0x0001;
static const int   MAX_VERTS = 2048;
static const int   MAX_BONES = 128;
static const int   MAX_TEX_SIZE = 512;
static const float PI_F = 3.14159265358979f;

struct V3 { float x, y, z; };
static V3 operator+(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static V3 operator-(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static V3 operator*(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static float Dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 Cross(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static float Len(V3 a) { return sqrtf(Dot(a, a)); }
static V3 Norm(V3 a, V3 fallback = { 0, 0, 1 })
{
    float l = Len(a);
    return l > 1e-12f ? a * (1.0f / l) : fallback;
}

struct M3 { float m[3][3]; };
static M3 M3Mul(const M3& a, const M3& b)
{
    M3 o;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            o.m[r][c] = a.m[r][0] * b.m[0][c] + a.m[r][1] * b.m[1][c] + a.m[r][2] * b.m[2][c];
    return o;
}
static V3 M3MulV(const M3& a, V3 v)
{
    return { a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
             a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
             a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z };
}
static M3 M3T(const M3& a)
{
    M3 o;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) o.m[r][c] = a.m[c][r];
    return o;
}
static M3 M3Identity() { return { { { 1, 0, 0 },{ 0, 1, 0 },{ 0, 0, 1 } } }; }

typedef std::array<float, 16> Mat4;
static Mat4 M4Identity() { Mat4 m = {}; m[0] = m[5] = m[10] = m[15] = 1; return m; }
static Mat4 M4Mul(const Mat4& a, const Mat4& b)
{
    Mat4 o;
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
        {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
            o[c * 4 + r] = s;
        }
    return o;
}
static V3 M4Point(const Mat4& m, V3 p)
{
    return { m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
             m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
             m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14] };
}
static V3 M4Dir(const Mat4& m, V3 p)
{
    return { m[0] * p.x + m[4] * p.y + m[8] * p.z,
             m[1] * p.x + m[5] * p.y + m[9] * p.z,
             m[2] * p.x + m[6] * p.y + m[10] * p.z };
}
static Mat4 M4AffineInverse(const Mat4& m)
{
    float a[3][3];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) a[r][c] = m[c * 4 + r];
    float det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
        a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
        a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if (fabsf(det) < 1e-20f) return M4Identity();
    float id = 1.0f / det, inv[3][3];
    inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * id;
    inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * id;
    inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * id;
    inv[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * id;
    inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * id;
    inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * id;
    inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * id;
    inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * id;
    inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * id;
    Mat4 o = M4Identity();
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) o[c * 4 + r] = inv[r][c];
    V3 t = { m[12], m[13], m[14] };
    o[12] = -(inv[0][0] * t.x + inv[0][1] * t.y + inv[0][2] * t.z);
    o[13] = -(inv[1][0] * t.x + inv[1][1] * t.y + inv[1][2] * t.z);
    o[14] = -(inv[2][0] * t.x + inv[2][1] * t.y + inv[2][2] * t.z);
    return o;
}

struct NodePose { V3 t; float q[4]; V3 s; };

static Mat4 PoseToMat(const NodePose& p)
{
    float x = p.q[0], y = p.q[1], z = p.q[2], w = p.q[3];
    float R[3][3] = {
        { 1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w) },
        { 2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w) },
        { 2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y) } };
    float sc[3] = { p.s.x, p.s.y, p.s.z };
    Mat4  m = M4Identity();
    for (int c = 0; c < 3; c++)
        for (int r = 0; r < 3; r++) m[c * 4 + r] = R[r][c] * sc[c];
    m[12] = p.t.x; m[13] = p.t.y; m[14] = p.t.z;
    return m;
}

static NodePose MatToPose(const Mat4& m)
{
    NodePose p;
    p.t = { m[12], m[13], m[14] };
    V3 c0 = { m[0], m[1], m[2] }, c1 = { m[4], m[5], m[6] }, c2 = { m[8], m[9], m[10] };
    float sx = Len(c0), sy = Len(c1), sz = Len(c2);
    if (Dot(Cross(c0, c1), c2) < 0) sx = -sx;
    p.s = { sx, sy, sz };
    c0 = c0 * (1.0f / (sx != 0 ? sx : 1)); c1 = c1 * (1.0f / (sy != 0 ? sy : 1)); c2 = c2 * (1.0f / (sz != 0 ? sz : 1));
    float r00 = c0.x, r10 = c0.y, r20 = c0.z, r01 = c1.x, r11 = c1.y, r21 = c1.z, r02 = c2.x, r12 = c2.y, r22 = c2.z;
    float tr = r00 + r11 + r22, x, y, z, w;
    if (tr > 0) { float s = sqrtf(tr + 1) * 2;               w = s / 4; x = (r21 - r12) / s; y = (r02 - r20) / s; z = (r10 - r01) / s; }
    else if (r00 > r11 && r00 > r22) { float s = sqrtf(1 + r00 - r11 - r22) * 2;  w = (r21 - r12) / s; x = s / 4; y = (r01 + r10) / s; z = (r02 + r20) / s; }
    else if (r11 > r22) { float s = sqrtf(1 + r11 - r00 - r22) * 2;  w = (r02 - r20) / s; x = (r01 + r10) / s; y = s / 4; z = (r12 + r21) / s; }
    else { float s = sqrtf(1 + r22 - r00 - r11) * 2;  w = (r10 - r01) / s; x = (r02 + r20) / s; y = (r12 + r21) / s; z = s / 4; }
    p.q[0] = x; p.q[1] = y; p.q[2] = z; p.q[3] = w;
    return p;
}

struct Rigid { M3 R; V3 t; };
static Rigid RigidOf(const Mat4& m)
{
    V3 c0 = Norm({ m[0], m[1], m[2] }, { 1, 0, 0 });
    V3 c1 = { m[4], m[5], m[6] };
    c1 = Norm(c1 - c0 * Dot(c1, c0), { 0, 1, 0 });
    V3 c2 = Cross(c0, c1);
    Rigid r;
    r.R = { { { c0.x, c1.x, c2.x },{ c0.y, c1.y, c2.y },{ c0.z, c1.z, c2.z } } };
    r.t = { m[12], m[13], m[14] };
    return r;
}

static const M3 PERM = { { { 0, 0, 1 },{ 1, 0, 0 },{ 0, 1, 0 } } };
static V3 ToGS(V3 v) { return { v.z, v.x, v.y }; }
static Rigid RigidToGS(const Rigid& r, float scale)
{
    Rigid o;
    o.R = M3Mul(M3Mul(PERM, r.R), M3T(PERM));
    o.t = ToGS(r.t) * scale;
    return o;
}

static void EulerFromR(const M3& R, float e[3])
{
    double r00 = R.m[0][0], r10 = R.m[1][0], r20 = R.m[2][0];
    double r21 = R.m[2][1], r22 = R.m[2][2], r11 = R.m[1][1], r12 = R.m[1][2];
    double cp = sqrt(r00 * r00 + r10 * r10);
    if (cp > 3e-4)
    {
        e[1] = (float)atan2(-r20, cp);
        e[0] = (float)atan2(r21, r22);
        e[2] = (float)atan2(r10, r00);
    }
    else
    {
        e[1] = (float)atan2(-r20, cp);
        e[2] = 0;
        e[0] = (float)atan2(-r12, r11);
    }
}

struct Buf
{
    std::vector<uint8_t> d;
    size_t Align() { while (d.size() % 4) d.push_back(0); return d.size(); }
    size_t PutRaw(const void* p, size_t n)
    {
        size_t o = d.size();
        if (n) d.insert(d.end(), (const uint8_t*)p, (const uint8_t*)p + n);
        return o;
    }
    template <class T> size_t Put(const T& v) { return PutRaw(&v, sizeof(T)); }
};

struct OutTex
{
    int                  w = 0, h = 0;
    bool                 masked = false;
    std::vector<uint8_t> pixels;
    uint8_t              pal[768] = {};
};

static void Quantize(const unsigned char* px, int w, int h, int maxColors, bool masked,
    std::vector<uint8_t>& out, uint8_t pal[768])
{
    struct Bin { int n = 0; double r = 0, g = 0, b = 0; };
    std::vector<Bin> bins(32768);

    auto keyOf = [](const unsigned char* p) { return ((p[0] >> 3) << 10) | ((p[1] >> 3) << 5) | (p[2] >> 3); };
    auto chan = [](int key, int axis) { return axis == 0 ? (key >> 10) : axis == 1 ? ((key >> 5) & 31) : (key & 31); };

    for (int i = 0; i < w * h; i++)
    {
        const unsigned char* p = px + i * 4;
        if (masked && p[3] < 128) continue;
        Bin& b = bins[keyOf(p)];
        b.n++; b.r += p[0]; b.g += p[1]; b.b += p[2];
    }

    std::vector<int> all;
    for (int k = 0; k < 32768; k++) if (bins[k].n) all.push_back(k);

    std::vector<std::vector<int>> boxes;
    if (!all.empty()) boxes.push_back(all);

    while ((int)boxes.size() < maxColors)
    {
        int best = -1, bestAxis = 0, bestRange = 0;
        for (size_t i = 0; i < boxes.size(); i++)
        {
            if (boxes[i].size() < 2) continue;
            for (int a = 0; a < 3; a++)
            {
                int lo = 31, hi = 0;
                for (int k : boxes[i]) { int c = chan(k, a); lo = std::min(lo, c); hi = std::max(hi, c); }
                if (hi - lo > bestRange) { bestRange = hi - lo; best = (int)i; bestAxis = a; }
            }
        }
        if (best < 0) break;

        std::vector<int> box = std::move(boxes[best]);
        std::sort(box.begin(), box.end(), [&](int a, int b) { return chan(a, bestAxis) < chan(b, bestAxis); });

        long total = 0;
        for (int k : box) total += bins[k].n;
        long   acc = 0;
        size_t split = 1;
        for (size_t i = 0; i < box.size(); i++)
        {
            acc += bins[box[i]].n;
            if (acc * 2 >= total) { split = i + 1; break; }
        }
        if (split >= box.size()) split = box.size() - 1;

        boxes[best].assign(box.begin(), box.begin() + split);
        boxes.emplace_back(box.begin() + split, box.end());
    }

    memset(pal, 0, 768);
    int nPal = (int)boxes.size();
    for (int i = 0; i < nPal; i++)
    {
        double n = 0, r = 0, g = 0, b = 0;
        for (int k : boxes[i]) { n += bins[k].n; r += bins[k].r; g += bins[k].g; b += bins[k].b; }
        pal[i * 3 + 0] = (uint8_t)(r / n + 0.5);
        pal[i * 3 + 1] = (uint8_t)(g / n + 0.5);
        pal[i * 3 + 2] = (uint8_t)(b / n + 0.5);
    }
    if (masked) { pal[255 * 3 + 0] = 0; pal[255 * 3 + 1] = 0; pal[255 * 3 + 2] = 255; }

    std::vector<int> lut(32768, -1);
    out.resize((size_t)w * h);
    for (int i = 0; i < w * h; i++)
    {
        const unsigned char* p = px + i * 4;
        if (masked && p[3] < 128) { out[i] = 255; continue; }
        int key = keyOf(p);
        if (lut[key] < 0)
        {
            const Bin& b = bins[key];
            double r = b.r / b.n, g = b.g / b.n, bl = b.b / b.n;
            double bestD = 1e18;
            int    bestI = 0;
            for (int c = 0; c < nPal; c++)
            {
                double dr = r - pal[c * 3], dg = g - pal[c * 3 + 1], db = bl - pal[c * 3 + 2];
                double dist = dr * dr + dg * dg + db * db;
                if (dist < bestD) { bestD = dist; bestI = c; }
            }
            lut[key] = bestI;
        }
        out[i] = (uint8_t)lut[key];
    }
}

static int PickSize(int v)
{
    int p = 16;
    while (p * 3 / 2 < v && p < MAX_TEX_SIZE) p <<= 1;
    return p;
}

static std::vector<uint8_t> ResizeRGBA(const uint8_t* src, int sw, int sh, int dw, int dh)
{
    std::vector<uint8_t> out((size_t)dw * dh * 4);
    for (int dy = 0; dy < dh; dy++)
    {
        double sy0 = (double)dy * sh / dh, sy1 = (double)(dy + 1) * sh / dh;
        for (int dx = 0; dx < dw; dx++)
        {
            double sx0 = (double)dx * sw / dw, sx1 = (double)(dx + 1) * sw / dw;
            double acc[4] = { 0, 0, 0, 0 }, wsum = 0;
            for (int y = (int)floor(sy0); y < (int)ceil(sy1) && y < sh; y++)
            {
                double wy = std::min<double>(y + 1, sy1) - std::max<double>(y, sy0);
                for (int x = (int)floor(sx0); x < (int)ceil(sx1) && x < sw; x++)
                {
                    double wx = std::min<double>(x + 1, sx1) - std::max<double>(x, sx0);
                    double wgt = wx * wy;
                    const uint8_t* p = src + ((size_t)y * sw + x) * 4;
                    for (int c = 0; c < 4; c++) acc[c] += p[c] * wgt;
                    wsum += wgt;
                }
            }
            uint8_t* o = &out[((size_t)dy * dw + dx) * 4];
            for (int c = 0; c < 4; c++) o[c] = (uint8_t)(wsum > 0 ? acc[c] / wsum + 0.5 : 255);
        }
    }
    return out;
}

static OutTex MakeTexture(const cg::cgltf_material* mat, const std::string& dir)
{
    OutTex t;
    float  factor[4] = { 1, 1, 1, 1 };
    bool   opaque = true;

    int            sw = 0, sh = 0, comp = 0;
    unsigned char* img = nullptr;

    if (mat)
    {
        if (mat->has_pbr_metallic_roughness)
        {
            for (int i = 0; i < 4; i++) factor[i] = mat->pbr_metallic_roughness.base_color_factor[i];
            const cg::cgltf_texture* tex = mat->pbr_metallic_roughness.base_color_texture.texture;
            const cg::cgltf_image* im = tex ? tex->image : nullptr;
            if (im)
            {
                if (im->buffer_view && im->buffer_view->buffer && im->buffer_view->buffer->data)
                {
                    const unsigned char* p = (const unsigned char*)im->buffer_view->buffer->data + im->buffer_view->offset;
                    img = stbi_load_from_memory(p, (int)im->buffer_view->size, &sw, &sh, &comp, 4);
                }
                else if (im->uri && strncmp(im->uri, "data:", 5) != 0)
                {
                    std::string u = im->uri;
                    cg::cgltf_decode_uri(&u[0]);
                    u.resize(strlen(u.c_str()));
                    img = stbi_load((dir + u).c_str(), &sw, &sh, &comp, 4);
                }
                if (!img) printf("Warning: could not decode the texture of material '%s'.\n", mat->name ? mat->name : "?");
            }
        }
        opaque = (mat->alpha_mode == cg::cgltf_alpha_mode_opaque);
    }

    std::vector<uint8_t> rgba;
    if (img)
    {
        rgba.assign(img, img + (size_t)sw * sh * 4);
        stbi_image_free(img);
    }
    else
    {
        sw = sh = 16;
        rgba.assign((size_t)sw * sh * 4, 255);
    }

    for (size_t i = 0; i < (size_t)sw * sh; i++)
        for (int c = 0; c < 4; c++)
        {
            float v = rgba[i * 4 + c] * factor[c];
            rgba[i * 4 + c] = (uint8_t)std::min(255.0f, std::max(0.0f, v + 0.5f));
        }
    if (opaque)
        for (size_t i = 0; i < (size_t)sw * sh; i++) rgba[i * 4 + 3] = 255;

    int w = PickSize(sw), h = PickSize(sh);
    if (w != sw || h != sh)
    {
        rgba = ResizeRGBA(rgba.data(), sw, sh, w, h);
        sw = w; sh = h;
    }

    for (int i = 0; i < w * h; i++)
        if (rgba[(size_t)i * 4 + 3] < 128) { t.masked = true; break; }

    t.w = w;
    t.h = h;
    Quantize(rgba.data(), w, h, t.masked ? 255 : 256, t.masked, t.pixels, t.pal);
    return t;
}

struct OutMesh
{
    std::vector<float>         verts, norms;
    std::vector<uint8_t>       vbone, nbone;
    std::vector<StudioTriVert> tris;
    int                        skinref = 0;
};

struct Bone
{
    int         node = -1;
    int         parent = -1;
    Rigid       bind;
    V3          pos = { 0, 0, 0 };
    float       eul[3] = { 0, 0, 0 };
    float       scale[6] = { 1, 1, 1, 1, 1, 1 };
    std::string name;
};

struct PoseV { float v[6]; };

struct Chan
{
    int                     node = -1;
    int                     path = 0;
    cg::cgltf_interpolation_type interp = cg::cgltf_interpolation_type_linear;
    int                     comps = 3;
    std::vector<float>      times, vals;
};

struct Anim
{
    std::string       name;
    std::vector<Chan> chans;
    float             duration = 0;
};

static void SampleChan(const Chan& c, float t, float* out)
{
    const int    C = c.comps;
    const size_t n = c.times.size();
    const bool   cubic = c.interp == cg::cgltf_interpolation_type_cubic_spline;
    auto val = [&](size_t k, int w) -> const float* { return cubic ? &c.vals[(k * 3 + w) * C] : &c.vals[k * C]; };

    if (n == 0) return;
    if (n == 1 || t <= c.times[0]) { memcpy(out, val(0, 1), C * sizeof(float)); return; }
    if (t >= c.times[n - 1]) { memcpy(out, val(n - 1, 1), C * sizeof(float)); return; }

    size_t k = (size_t)(std::upper_bound(c.times.begin(), c.times.end(), t) - c.times.begin()) - 1;
    float  dt = c.times[k + 1] - c.times[k];
    float  s = dt > 0 ? (t - c.times[k]) / dt : 0;

    if (c.interp == cg::cgltf_interpolation_type_step) { memcpy(out, val(k, 1), C * sizeof(float)); return; }

    if (cubic)
    {
        const float* p0 = val(k, 1), * m0 = val(k, 2), * p1 = val(k + 1, 1), * m1 = val(k + 1, 0);
        float s2 = s * s, s3 = s2 * s;
        float h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s, h01 = -2 * s3 + 3 * s2, h11 = s3 - s2;
        for (int i = 0; i < C; i++) out[i] = h00 * p0[i] + h10 * dt * m0[i] + h01 * p1[i] + h11 * dt * m1[i];
    }
    else
    {
        const float* a = val(k, 1), * b = val(k + 1, 1);
        if (C == 4)
        {
            float bb[4] = { b[0], b[1], b[2], b[3] };
            float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
            if (d < 0) { for (int i = 0; i < 4; i++) bb[i] = -bb[i]; d = -d; }
            float wa, wb;
            if (d > 0.9995f) { wa = 1 - s; wb = s; }
            else
            {
                float th = acosf(d), sn = sinf(th);
                wa = sinf((1 - s) * th) / sn;
                wb = sinf(s * th) / sn;
            }
            for (int i = 0; i < 4; i++) out[i] = wa * a[i] + wb * bb[i];
        }
        else
            for (int i = 0; i < C; i++) out[i] = a[i] + (b[i] - a[i]) * s;
    }
    if (C == 4)
    {
        float l = sqrtf(out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]);
        if (l > 1e-12f) for (int i = 0; i < 4; i++) out[i] /= l;
    }
}

static std::string SafeName(const char* s, const char* fallback, size_t maxLen)
{
    std::string r = (s && *s) ? s : fallback;
    for (char& c : r) if ((unsigned char)c < 32 || (unsigned char)c > 126) c = '_';
    if (r.size() > maxLen) r.resize(maxLen);
    return r;
}

static void EncodeChannel(const std::vector<short>& q, std::vector<uint8_t>& out)
{
    size_t n = q.size();
    std::vector<int> rl(n, 1);
    for (size_t k = n; k-- > 1;)
        rl[k - 1] = (q[k - 1] == q[k]) ? rl[k] + 1 : 1;

    auto putShort = [&](short v) { out.push_back((uint8_t)(v & 0xFF)); out.push_back((uint8_t)((v >> 8) & 0xFF)); };

    size_t i = 0;
    while (i < n)
    {
        if (rl[i] >= 3)
        {
            int cnt = std::min(rl[i], 255);
            out.push_back(1); out.push_back((uint8_t)cnt);
            putShort(q[i]);
            i += cnt;
        }
        else
        {
            size_t j = i + 1;
            while (j < n && (j - i) < 255 && rl[j] < 3) j++;
            out.push_back((uint8_t)(j - i)); out.push_back((uint8_t)(j - i));
            for (size_t k = i; k < j; k++) putShort(q[k]);
            i = j;
        }
    }

    out.push_back(1); out.push_back(1);
    putShort(n ? q[n - 1] : 0);
}

struct Opts
{
    std::string in, out;
    float       scale = 40.0f, fps = 30.0f, height = 0.0f;
    bool        flip = true;
};

static M3 EulerToR(const float e[3])
{
    double sy = sin(e[2] * 0.5), cy = cos(e[2] * 0.5);
    double sp = sin(e[1] * 0.5), cp = cos(e[1] * 0.5);
    double sr = sin(e[0] * 0.5), cr = cos(e[0] * 0.5);
    double x = sr * cp * cy - cr * sp * sy, y = cr * sp * cy + sr * cp * sy;
    double z = cr * cp * sy - sr * sp * cy, w = cr * cp * cy + sr * sp * sy;
    M3 R;
    R.m[0][0] = (float)(1 - 2 * y * y - 2 * z * z); R.m[0][1] = (float)(2 * x * y - 2 * w * z);     R.m[0][2] = (float)(2 * x * z + 2 * w * y);
    R.m[1][0] = (float)(2 * x * y + 2 * w * z);     R.m[1][1] = (float)(1 - 2 * x * x - 2 * z * z); R.m[1][2] = (float)(2 * y * z - 2 * w * x);
    R.m[2][0] = (float)(2 * x * z - 2 * w * y);     R.m[2][1] = (float)(2 * y * z + 2 * w * x);     R.m[2][2] = (float)(1 - 2 * x * x - 2 * y * y);
    return R;
}

static int Convert(const Opts& o0, float scale, float* measureHeight)
{
    const std::string in = o0.in;
    std::string       out = o0.out;
    const float       fps = o0.fps;
    const bool        flip = o0.flip;

    if (out.empty())
    {
        out = in;
        size_t dot = out.find_last_of('.');
        size_t sl = out.find_last_of("/\\");
        if (dot != std::string::npos && (sl == std::string::npos || dot > sl)) out.erase(dot);
        out += ".mdl";
    }
    std::string dir;
    {
        size_t sl = in.find_last_of("/\\");
        if (sl != std::string::npos) dir = in.substr(0, sl + 1);
    }

    cg::cgltf_options opt = {};
    cg::cgltf_data* data = nullptr;
    if (cg::cgltf_parse_file(&opt, in.c_str(), &data) != cg::cgltf_result_success)
    {
        printf("Error: could not open/read '%s'.\n", in.c_str());
        return 2;
    }
    if (cg::cgltf_load_buffers(&opt, data, in.c_str()) != cg::cgltf_result_success)
    {
        printf("Error: could not load the buffers of '%s'.\n", in.c_str());
        cg::cgltf_free(data);
        return 2;
    }

    const size_t nNodes = data->nodes_count;
    auto nodeIdx = [&](const cg::cgltf_node* n) { return (int)(n - data->nodes); };

    std::vector<NodePose> defPose(nNodes);
    for (size_t i = 0; i < nNodes; i++)
    {
        const cg::cgltf_node& n = data->nodes[i];
        if (n.has_matrix)
        {
            Mat4 m;
            for (int k = 0; k < 16; k++) m[k] = n.matrix[k];
            defPose[i] = MatToPose(m);
        }
        else
        {
            defPose[i].t = n.has_translation ? V3{ n.translation[0], n.translation[1], n.translation[2] } : V3{ 0, 0, 0 };
            if (n.has_rotation) for (int k = 0; k < 4; k++) defPose[i].q[k] = n.rotation[k];
            else { defPose[i].q[0] = defPose[i].q[1] = defPose[i].q[2] = 0; defPose[i].q[3] = 1; }
            defPose[i].s = n.has_scale ? V3{ n.scale[0], n.scale[1], n.scale[2] } : V3{ 1, 1, 1 };
        }
    }

    auto computeWorld = [&](const std::vector<NodePose>& poses, std::vector<Mat4>& world) {
        world.assign(nNodes, M4Identity());
        std::vector<char> done(nNodes, 0);
        for (size_t i = 0; i < nNodes; i++)
        {

            std::vector<int> chain;
            int cur = (int)i;
            while (cur >= 0 && !done[cur])
            {
                chain.push_back(cur);
                const cg::cgltf_node* p = data->nodes[cur].parent;
                cur = p ? nodeIdx(p) : -1;
            }
            for (size_t k = chain.size(); k-- > 0;)
            {
                int  n = chain[k];
                Mat4 local = PoseToMat(poses[n]);
                const cg::cgltf_node* p = data->nodes[n].parent;
                world[n] = p ? M4Mul(world[nodeIdx(p)], local) : local;
                done[n] = 1;
            }
        }
        };

    std::vector<Mat4> W0;
    computeWorld(defPose, W0);

    std::vector<char>              isJoint(nNodes, 0);
    std::vector<std::vector<Mat4>> skinIbm(data->skins_count);
    size_t                         jointCount = 0;
    for (size_t s = 0; s < data->skins_count; s++)
    {
        const cg::cgltf_skin& sk = data->skins[s];
        skinIbm[s].assign(sk.joints_count, M4Identity());
        for (size_t j = 0; j < sk.joints_count; j++)
        {
            int n = nodeIdx(sk.joints[j]);
            if (!isJoint[n]) { isJoint[n] = 1; jointCount++; }
            if (sk.inverse_bind_matrices)
            {
                float f[16];
                if (cg::cgltf_accessor_read_float(sk.inverse_bind_matrices, j, f, 16))
                    for (int k = 0; k < 16; k++) skinIbm[s][j][k] = f[k];
            }
        }
    }

    auto jointParent = [&](int n) -> int {
        const cg::cgltf_node* p = data->nodes[n].parent;
        while (p)
        {
            int pi = nodeIdx(p);
            if (isJoint[pi]) return pi;
            p = p->parent;
        }
        return -1;
        };

    std::vector<Bone> bones;
    std::vector<int>  nodeBone(nNodes, -1);

    if (jointCount == 0)
    {
        Bone b;
        b.name = "root";
        b.bind.R = M3Identity();
        b.bind.t = { 0, 0, 0 };
        bones.push_back(b);
    }
    else
    {
        std::vector<int>  order;
        std::vector<char> placed(nNodes, 0);
        bool              progress = true;
        while (order.size() < jointCount && progress)
        {
            progress = false;
            for (size_t n = 0; n < nNodes; n++)
            {
                if (!isJoint[n] || placed[n]) continue;
                int jp = jointParent((int)n);
                if (jp < 0 || placed[jp]) { order.push_back((int)n); placed[n] = 1; progress = true; }
            }
        }
        if (order.size() > (size_t)MAX_BONES)
        {
            printf("Error: %zu bones (GoldSrc limit: %d). Reduce the skeleton in Blender.\n", order.size(), MAX_BONES);
            cg::cgltf_free(data);
            return 4;
        }
        for (int n : order)
        {
            Bone b;
            b.node = n;
            b.name = SafeName(data->nodes[n].name, ("bone" + std::to_string(bones.size())).c_str(), 31);
            b.bind = RigidToGS(RigidOf(W0[n]), scale);
            nodeBone[n] = (int)bones.size();
            bones.push_back(b);
        }
        for (Bone& b : bones)
        {
            int jp = jointParent(b.node);
            b.parent = jp >= 0 ? nodeBone[jp] : -1;
        }
    }

    auto localOf = [&](const Rigid& self, const Rigid* parent, float v[6]) {
        M3 Rl = self.R;
        V3 tl = self.t;
        if (parent)
        {
            M3 Rpt = M3T(parent->R);
            Rl = M3Mul(Rpt, self.R);
            tl = M3MulV(Rpt, self.t - parent->t);
        }
        v[0] = tl.x; v[1] = tl.y; v[2] = tl.z;
        EulerFromR(Rl, v + 3);
        };
    for (Bone& b : bones)
    {
        float v[6];
        localOf(b.bind, b.parent >= 0 ? &bones[b.parent].bind : nullptr, v);
        b.pos = { v[0], v[1], v[2] };
        b.eul[0] = v[3]; b.eul[1] = v[4]; b.eul[2] = v[5];
    }

    std::vector<OutTex> texs;
    for (size_t i = 0; i < data->materials_count; i++) texs.push_back(measureHeight ? OutTex() : MakeTexture(&data->materials[i], dir));
    int defaultTex = -1;

    std::vector<OutMesh> meshes;
    for (size_t ni = 0; ni < nNodes; ni++)
    {
        const cg::cgltf_node& node = data->nodes[ni];
        if (!node.mesh) continue;

        for (size_t pi = 0; pi < node.mesh->primitives_count; pi++)
        {
            const cg::cgltf_primitive& prim = node.mesh->primitives[pi];
            if (prim.type != cg::cgltf_primitive_type_triangles)
            {
                printf("Warning: non-triangle primitive ignored (node '%s').\n", node.name ? node.name : "?");
                continue;
            }

            const cg::cgltf_accessor* aPos = nullptr, * aNor = nullptr, * aUv = nullptr, * aJoints = nullptr, * aWeights = nullptr;
            for (size_t ai = 0; ai < prim.attributes_count; ai++)
            {
                const cg::cgltf_attribute& at = prim.attributes[ai];
                if (at.type == cg::cgltf_attribute_type_position)                       aPos = at.data;
                else if (at.type == cg::cgltf_attribute_type_normal)                    aNor = at.data;
                else if (at.type == cg::cgltf_attribute_type_texcoord && at.index == 0) aUv = at.data;
                else if (at.type == cg::cgltf_attribute_type_joints && at.index == 0)   aJoints = at.data;
                else if (at.type == cg::cgltf_attribute_type_weights && at.index == 0)  aWeights = at.data;
            }
            if (!aPos) continue;

            const bool skinned = node.skin && aJoints;
            const size_t vc = aPos->count;

            int texIdx;
            if (prim.material) texIdx = (int)(prim.material - data->materials);
            else
            {
                if (defaultTex < 0) { defaultTex = (int)texs.size(); texs.push_back(measureHeight ? OutTex() : MakeTexture(nullptr, dir)); }
                texIdx = defaultTex;
            }
            const OutTex& tex = texs[texIdx];

            struct SrcV { V3 pw, nw; int bone; float u, v; };
            std::vector<SrcV> sv(vc);

            const int skinIdx = node.skin ? (int)(node.skin - data->skins) : -1;
            int unskinnedBone = 0;
            if (!skinned && jointCount > 0)
            {
                int cur = (int)ni;
                const cg::cgltf_node* p = &node;
                while (p) { cur = nodeIdx(p); if (isJoint[cur]) { unskinnedBone = nodeBone[cur]; break; } p = p->parent; }
            }

            for (size_t i = 0; i < vc; i++)
            {
                float p[3] = { 0, 0, 0 }, n[3] = { 0, 0, 1 }, uv[2] = { 0, 0 };
                cg::cgltf_accessor_read_float(aPos, i, p, 3);
                bool hasN = aNor && cg::cgltf_accessor_read_float(aNor, i, n, 3);
                if (aUv) cg::cgltf_accessor_read_float(aUv, i, uv, 2);

                V3 pos = { p[0], p[1], p[2] }, nor = { n[0], n[1], n[2] };
                SrcV s;
                if (skinned)
                {
                    cg::cgltf_uint j[4] = { 0, 0, 0, 0 };
                    float          w[4] = { 1, 0, 0, 0 };
                    cg::cgltf_accessor_read_uint(aJoints, i, j, 4);
                    if (aWeights) cg::cgltf_accessor_read_float(aWeights, i, w, 4);
                    int best = 0;
                    for (int k = 1; k < 4; k++) if (w[k] > w[best]) best = k;
                    size_t ji = std::min<size_t>(j[best], node.skin->joints_count - 1);
                    int    jn = nodeIdx(node.skin->joints[ji]);
                    int    bn = nodeBone[jn];
                    s.bone = bn >= 0 ? bn : 0;
                    Mat4 M = M4Mul(W0[jn], skinIbm[skinIdx][ji]);
                    pos = M4Point(M, pos);
                    nor = M4Dir(M, nor);
                }
                else
                {
                    pos = M4Point(W0[ni], pos);
                    nor = M4Dir(W0[ni], nor);
                    s.bone = unskinnedBone;
                }
                s.pw = ToGS(pos) * scale;
                s.nw = hasN ? Norm(ToGS(nor)) : V3{ 0, 0, 0 };
                s.u = uv[0];
                s.v = uv[1];
                sv[i] = s;
            }

            OutMesh cur;
            cur.skinref = texIdx;
            std::map<std::array<int, 4>, int> vmap, nmap;

            auto flush = [&]() {
                if (!cur.tris.empty()) meshes.push_back(std::move(cur));
                cur = OutMesh();
                cur.skinref = texIdx;
                vmap.clear();
                nmap.clear();
                };
            auto addV = [&](const std::array<int, 4>& k, V3 local, int bone) -> int {
                auto it = vmap.find(k);
                if (it != vmap.end()) return it->second;
                int idx = (int)cur.verts.size() / 3;
                cur.verts.push_back(local.x); cur.verts.push_back(local.y); cur.verts.push_back(local.z);
                cur.vbone.push_back((uint8_t)bone);
                vmap[k] = idx;
                return idx;
                };
            auto addN = [&](const std::array<int, 4>& k, V3 local, int bone) -> int {
                auto it = nmap.find(k);
                if (it != nmap.end()) return it->second;
                int idx = (int)cur.norms.size() / 3;
                cur.norms.push_back(local.x); cur.norms.push_back(local.y); cur.norms.push_back(local.z);
                cur.nbone.push_back((uint8_t)bone);
                nmap[k] = idx;
                return idx;
                };

            size_t ic = prim.indices ? prim.indices->count : vc;
            for (size_t t = 0; t + 2 < ic; t += 3)
            {
                size_t id[3];
                for (int k = 0; k < 3; k++)
                    id[k] = prim.indices ? cg::cgltf_accessor_read_index(prim.indices, t + k) : t + k;
                if (id[0] >= vc || id[1] >= vc || id[2] >= vc) continue;

                V3 fn = Norm(Cross(sv[id[1]].pw - sv[id[0]].pw, sv[id[2]].pw - sv[id[0]].pw));

                struct Corner { V3 lp, ln; int bone; short s, t; std::array<int, 4> kv, kn; };
                Corner cn[3];
                const int order[3] = { 0, flip ? 2 : 1, flip ? 1 : 2 };
                for (int oi = 0; oi < 3; oi++)
                {
                    const SrcV& sc = sv[id[order[oi]]];
                    const Rigid& bb = bones[sc.bone].bind;
                    M3 Rt = M3T(bb.R);
                    Corner& c = cn[oi];
                    c.lp = M3MulV(Rt, sc.pw - bb.t);
                    V3 nw = (Len(sc.nw) > 0.5f) ? sc.nw : fn;
                    c.ln = Norm(M3MulV(Rt, nw));
                    c.bone = sc.bone;
                    c.s = (short)lroundf(sc.u * tex.w);
                    c.t = (short)lroundf(sc.v * tex.h);
                    c.kv = { (int)lroundf(c.lp.x * 1000), (int)lroundf(c.lp.y * 1000), (int)lroundf(c.lp.z * 1000), sc.bone };
                    c.kn = { (int)lroundf(c.ln.x * 1000), (int)lroundf(c.ln.y * 1000), (int)lroundf(c.ln.z * 1000), sc.bone };
                }

                int newV = 0, newN = 0;
                for (int k = 0; k < 3; k++)
                {
                    bool dv = false, dn = false;
                    for (int j2 = 0; j2 < k; j2++) { dv |= (cn[j2].kv == cn[k].kv); dn |= (cn[j2].kn == cn[k].kn); }
                    if (!dv && !vmap.count(cn[k].kv)) newV++;
                    if (!dn && !nmap.count(cn[k].kn)) newN++;
                }
                if ((int)cur.verts.size() / 3 + newV > MAX_VERTS || (int)cur.norms.size() / 3 + newN > MAX_VERTS) flush();

                for (int k = 0; k < 3; k++)
                {
                    StudioTriVert tv;
                    tv.vertindex = (short)addV(cn[k].kv, cn[k].lp, cn[k].bone);
                    tv.normindex = (short)addN(cn[k].kn, cn[k].ln, cn[k].bone);
                    tv.s = cn[k].s;
                    tv.t = cn[k].t;
                    cur.tris.push_back(tv);
                }
            }
            flush();
        }
    }
    if (meshes.empty())
    {
        printf("Error: the file has no usable geometry.\n");
        cg::cgltf_free(data);
        return 3;
    }
    if (texs.size() > 100) printf("Warning: %zu textures (engine limit: 100).\n", texs.size());

    float bmin[3] = { 1e30f, 1e30f, 1e30f }, bmax[3] = { -1e30f, -1e30f, -1e30f };
    struct BV { V3 p; int bone; };
    std::vector<BV> allV;
    for (auto& m : meshes)
        for (size_t i = 0; i + 2 < m.verts.size(); i += 3)
            allV.push_back({ { m.verts[i], m.verts[i + 1], m.verts[i + 2] }, (int)m.vbone[i / 3] });

    auto extend = [&](const std::vector<Rigid>& worlds, float mn[3], float mx[3]) {
        for (const BV& v : allV)
        {
            const Rigid& r = worlds[v.bone];
            V3 w = M3MulV(r.R, v.p) + r.t;
            float c[3] = { w.x, w.y, w.z };
            for (int a = 0; a < 3; a++) { mn[a] = std::min(mn[a], c[a]); mx[a] = std::max(mx[a], c[a]); }
        }
        };
    {
        std::vector<Rigid> bindW;
        for (const Bone& bn : bones) bindW.push_back(bn.bind);
        extend(bindW, bmin, bmax);
    }
    if (measureHeight)
    {
        *measureHeight = bmax[2] - bmin[2];
        cg::cgltf_free(data);
        return 0;
    }

    std::vector<Anim> anims;
    for (size_t ai = 0; ai < data->animations_count; ai++)
    {
        const cg::cgltf_animation& a = data->animations[ai];
        Anim an;
        an.name = SafeName(a.name, ("anim" + std::to_string(ai)).c_str(), 31);
        for (size_t ci = 0; ci < a.channels_count; ci++)
        {
            const cg::cgltf_animation_channel& ch = a.channels[ci];
            if (!ch.target_node || !ch.sampler) continue;
            Chan c;
            c.node = nodeIdx(ch.target_node);
            switch (ch.target_path)
            {
            case cg::cgltf_animation_path_type_translation: c.path = 0; c.comps = 3; break;
            case cg::cgltf_animation_path_type_rotation:    c.path = 1; c.comps = 4; break;
            case cg::cgltf_animation_path_type_scale:       c.path = 2; c.comps = 3; break;
            default: continue;
            }
            c.interp = ch.sampler->interpolation;
            size_t nt = cg::cgltf_accessor_unpack_floats(ch.sampler->input, nullptr, 0);
            c.times.resize(nt);
            cg::cgltf_accessor_unpack_floats(ch.sampler->input, c.times.data(), nt);
            size_t nv = cg::cgltf_accessor_unpack_floats(ch.sampler->output, nullptr, 0);
            c.vals.resize(nv);
            cg::cgltf_accessor_unpack_floats(ch.sampler->output, c.vals.data(), nv);
            size_t need = nt * c.comps * (c.interp == cg::cgltf_interpolation_type_cubic_spline ? 3 : 1);
            if (nt == 0 || nv < need) continue;
            if (c.times.back() > an.duration) an.duration = c.times.back();
            an.chans.push_back(std::move(c));
        }
        if (!an.chans.empty()) anims.push_back(std::move(an));
    }

    const int nb = (int)bones.size();

    auto sampleAnim = [&](const Anim& an, float sfps, std::vector<std::vector<PoseV>>& poses) {
        int nf = std::max(1, (int)floorf(an.duration * sfps + 0.5f) + 1);
        poses.assign(nf, std::vector<PoseV>(nb));
        std::vector<Mat4> W;
        for (int f = 0; f < nf; f++)
        {
            float t = std::min(an.duration, f / sfps);
            std::vector<NodePose> np = defPose;
            for (const Chan& c : an.chans)
            {
                float v[4] = { 0, 0, 0, 1 };
                SampleChan(c, t, v);
                NodePose& p = np[c.node];
                if (c.path == 0)      p.t = { v[0], v[1], v[2] };
                else if (c.path == 1) { for (int k = 0; k < 4; k++) p.q[k] = v[k]; }
                else                  p.s = { v[0], v[1], v[2] };
            }
            computeWorld(np, W);

            std::vector<Rigid> world(nb);
            for (int b = 0; b < nb; b++)
                world[b] = bones[b].node >= 0 ? RigidToGS(RigidOf(W[bones[b].node]), scale) : bones[b].bind;

            for (int b = 0; b < nb; b++)
            {
                float v[6];
                localOf(world[b], bones[b].parent >= 0 ? &world[bones[b].parent] : nullptr, v);

                for (int k = 3; k < 6; k++)
                {
                    float prev = f > 0 ? poses[f - 1][b].v[k] : bones[b].eul[k - 3];
                    while (v[k] - prev > PI_F)  v[k] -= 2 * PI_F;
                    while (v[k] - prev < -PI_F) v[k] += 2 * PI_F;
                }
                memcpy(poses[f][b].v, v, sizeof(v));
            }
        }
        };

    auto defaultOf = [&](int b, int ch) { return ch < 3 ? (&bones[b].pos.x)[ch] : bones[b].eul[ch - 3]; };

    {
        std::vector<std::array<float, 6>> maxd(nb);
        for (auto& m : maxd) m.fill(0);
        for (const Anim& an : anims)
        {
            std::vector<std::vector<PoseV>> poses;
            sampleAnim(an, fps, poses);
            for (auto& fr : poses)
                for (int b = 0; b < nb; b++)
                    for (int c = 0; c < 6; c++)
                        maxd[b][c] = std::max(maxd[b][c], fabsf(fr[b].v[c] - defaultOf(b, c)));
        }
        for (int b = 0; b < nb; b++)
            for (int c = 0; c < 6; c++)
                bones[b].scale[c] = std::max(maxd[b][c] / 30000.0f * 1.001f, c < 3 ? 1e-4f : 1e-6f);
    }

    std::vector<std::vector<int>> groups;
    {
        int curV = 0, curN = 0;
        for (int i = 0; i < (int)meshes.size(); i++)
        {
            int nv = (int)meshes[i].verts.size() / 3, nn = (int)meshes[i].norms.size() / 3;
            if (nv > MAX_VERTS || nn > MAX_VERTS)
                printf("Warning: a mesh has %d vertices / %d normals (engine limit: %d). Simplify it.\n", nv, nn, MAX_VERTS);
            if (groups.empty() || curV + nv > MAX_VERTS || curN + nn > MAX_VERTS)
            {
                groups.emplace_back();
                curV = curN = 0;
            }
            groups.back().push_back(i);
            curV += nv;
            curN += nn;
        }
    }
    if (groups.size() > 32) printf("Warning: %zu bodyparts (engine limit: 32).\n", groups.size());

    {
        int totTris = 0, maxSubV = 0, maxSubN = 0, maxSubMesh = 0, maxTex = 0;
        for (auto& g : groups)
        {
            int sv2 = 0, sn2 = 0;
            for (int mi : g) { sv2 += (int)meshes[mi].verts.size() / 3; sn2 += (int)meshes[mi].norms.size() / 3; }
            maxSubV = std::max(maxSubV, sv2); maxSubN = std::max(maxSubN, sn2);
            maxSubMesh = std::max(maxSubMesh, (int)g.size());
        }
        for (auto& m : meshes) totTris += (int)m.tris.size() / 3;
        for (auto& t : texs) maxTex = std::max(maxTex, std::max(t.w, t.h));
        bool bad = false;
        auto line = [&](const char* what, int val, int lim) {
            bool over = val > lim;
            bad |= over;
            printf("    %s %-26s %6d / %d\n", over ? "[!]" : "   ", what, val, lim);
            };
        printf("GoldSrc engine limits:\n");
        line("bones", nb, 128);
        line("triangles (total)", totTris, 20000);
        line("bodyparts", (int)groups.size(), 32);
        line("vertices per submodel", maxSubV, 2048);
        line("normals per submodel", maxSubN, 2048);
        line("meshes per submodel", maxSubMesh, 256);
        line("textures", (int)texs.size(), 100);
        line("max texture size", maxTex, 512);
        line("sequences", (int)std::max<size_t>(1, anims.size()), 2048);
        if (bad)
            printf("    [!] WARNING: values above the engine limit. The model opens in the Model Viewer, but the game\n"
                "        may crash. Reduce polygons/bones/textures in Blender and convert again.\n");
    }

    Buf b;
    b.d.assign(sizeof(StudioHdr), 0);

    size_t boneOff = b.Align();
    for (const Bone& bn : bones)
    {
        StudioBone sb = {};
        strncpy(sb.name, bn.name.c_str(), 31);
        sb.parent = bn.parent;
        for (int k = 0; k < 6; k++) sb.bonecontroller[k] = -1;
        sb.value[0] = bn.pos.x; sb.value[1] = bn.pos.y; sb.value[2] = bn.pos.z;
        sb.value[3] = bn.eul[0]; sb.value[4] = bn.eul[1]; sb.value[5] = bn.eul[2];
        for (int k = 0; k < 6; k++) sb.scale[k] = bn.scale[k];
        b.Put(sb);
    }

    StudioSeqGroup sg = {};
    strcpy(sg.label, "default");
    size_t sgOff = b.Align();
    b.Put(sg);

    struct SeqOut { std::string name; float fps; int frames; int flags; size_t animOff; float bmin[3], bmax[3]; };
    std::vector<SeqOut> seqs;
    int totalFrames = 0;
    float gmin[3] = { bmin[0], bmin[1], bmin[2] }, gmax[3] = { bmax[0], bmax[1], bmax[2] };

    auto writeSeq = [&](const std::string& name, const std::vector<std::vector<PoseV>>* poses, float sfps) -> bool {
        int nf = poses ? (int)poses->size() : 1;
        size_t baseOff = b.Align();

        std::vector<uint8_t> blk(12 * (size_t)nb, 0);
        bool ok = true;
        if (poses)
        {
            for (int bi = 0; bi < nb && ok; bi++)
            {
                for (int ch = 0; ch < 6; ch++)
                {
                    float def = defaultOf(bi, ch), maxd = 0;
                    for (int f = 0; f < nf; f++) maxd = std::max(maxd, fabsf((*poses)[f][bi].v[ch] - def));
                    if (maxd < (ch < 3 ? 1e-3f : 1e-5f)) continue;

                    std::vector<short> q(nf);
                    for (int f = 0; f < nf; f++)
                    {
                        float qv = ((*poses)[f][bi].v[ch] - def) / bones[bi].scale[ch];
                        q[f] = (short)std::max(-32767.0f, std::min(32767.0f, floorf(qv + 0.5f)));
                    }
                    size_t dataPos = blk.size();
                    size_t rel = dataPos - (size_t)bi * 12;
                    if (rel > 65535) { ok = false; break; }
                    uint16_t off = (uint16_t)rel;
                    memcpy(&blk[(size_t)bi * 12 + ch * 2], &off, 2);
                    EncodeChannel(q, blk);
                }
            }
        }
        if (!ok) return false;

        b.PutRaw(blk.data(), blk.size());

        int flags = 0;
        if (poses && nf > 1)
        {
            float dp = 0, dr = 0;
            for (int bi = 0; bi < nb; bi++)
                for (int ch = 0; ch < 6; ch++)
                {
                    float d = fabsf((*poses)[0][bi].v[ch] - (*poses)[nf - 1][bi].v[ch]);
                    if (ch < 3) dp = std::max(dp, d); else dr = std::max(dr, d);
                }
            if (dp < 0.05f && dr < 0.002f) flags |= STUDIO_LOOPING;
        }
        SeqOut so = { name, sfps, nf, flags, baseOff,{ 1e30f, 1e30f, 1e30f },{ -1e30f, -1e30f, -1e30f } };
        if (poses)
        {
            for (int f = 0; f < nf; f++)
            {
                std::vector<Rigid> w(nb);
                for (int bi = 0; bi < nb; bi++)
                {
                    const float* v = (*poses)[f][bi].v;
                    M3 Rl = EulerToR(v + 3);
                    V3 tl = { v[0], v[1], v[2] };
                    int p = bones[bi].parent;
                    if (p >= 0) { w[bi].R = M3Mul(w[p].R, Rl); w[bi].t = M3MulV(w[p].R, tl) + w[p].t; }
                    else { w[bi].R = Rl; w[bi].t = tl; }
                }
                extend(w, so.bmin, so.bmax);
            }
        }
        else
        {
            memcpy(so.bmin, bmin, sizeof(bmin));
            memcpy(so.bmax, bmax, sizeof(bmax));
        }
        for (int a = 0; a < 3; a++) { gmin[a] = std::min(gmin[a], so.bmin[a]); gmax[a] = std::max(gmax[a], so.bmax[a]); }
        seqs.push_back(so);
        totalFrames += nf;
        return true;
        };

    if (anims.empty())
    {
        writeSeq("idle", nullptr, fps);
    }
    else
    {
        for (const Anim& an : anims)
        {
            float sfps = fps;
            bool  done = false;
            for (int attempt = 0; attempt < 10 && !done; attempt++)
            {
                std::vector<std::vector<PoseV>> poses;
                sampleAnim(an, sfps, poses);
                done = writeSeq(an.name, &poses, sfps);
                if (!done)
                {
                    printf("Warning: '%s' is too large at %.1f fps, reducing...\n", an.name.c_str(), sfps);
                    sfps *= 0.7f;
                    if (sfps < 2) break;
                }
            }
            if (!done) printf("Error: could not encode animation '%s'.\n", an.name.c_str());
        }
        if (seqs.empty()) writeSeq("idle", nullptr, fps);
    }

    size_t seqOff = b.Align();
    for (const SeqOut& s : seqs)
    {
        StudioSeqDesc sq = {};
        strncpy(sq.label, s.name.c_str(), 31);
        sq.fps = s.fps;
        sq.flags = s.flags;
        sq.actweight = 0;
        sq.numframes = s.frames;
        sq.numblends = 1;
        sq.animindex = (int)s.animOff;
        sq.blendend[0] = 1.0f;
        memcpy(sq.bbmin, s.bmin, sizeof(s.bmin));
        memcpy(sq.bbmax, s.bmax, sizeof(s.bmax));
        b.Put(sq);
    }

    std::vector<StudioTexture> st(texs.size());
    for (size_t i = 0; i < texs.size(); i++)
    {
        st[i] = {};
        snprintf(st[i].name, sizeof(st[i].name), "mat%zu.bmp", i);
        st[i].flags = texs[i].masked ? STUDIO_NF_MASKED : 0;
        st[i].width = texs[i].w;
        st[i].height = texs[i].h;
        st[i].index = 0;
    }
    size_t texOff = b.Align();
    for (auto& t : st) b.Put(t);

    size_t skinOff = b.Align();
    for (size_t i = 0; i < texs.size(); i++) b.Put((short)i);

    struct HBox { int bone, group; float bbmin[3], bbmax[3]; };
    static_assert(sizeof(HBox) == 32, "mstudiobbox_t");
    std::vector<HBox> hboxes;
    {
        std::vector<HBox> hb(nb);
        std::vector<char> used(nb, 0);
        for (int bi = 0; bi < nb; bi++)
        {
            hb[bi].bone = bi; hb[bi].group = 0;
            for (int a = 0; a < 3; a++) { hb[bi].bbmin[a] = 1e30f; hb[bi].bbmax[a] = -1e30f; }
        }
        for (const BV& v : allV)
        {
            float c[3] = { v.p.x, v.p.y, v.p.z };
            used[v.bone] = 1;
            for (int a = 0; a < 3; a++)
            {
                hb[v.bone].bbmin[a] = std::min(hb[v.bone].bbmin[a], c[a]);
                hb[v.bone].bbmax[a] = std::max(hb[v.bone].bbmax[a], c[a]);
            }
        }
        for (int bi = 0; bi < nb; bi++) if (used[bi]) hboxes.push_back(hb[bi]);
    }
    size_t hbOff = b.Align();
    for (const HBox& hbx : hboxes) b.Put(hbx);

    std::vector<StudioBodypart> bodyparts;
    for (size_t gi = 0; gi < groups.size(); gi++)
    {
        std::vector<float>      verts, norms;
        std::vector<uint8_t>    vbone, nbone;
        std::vector<StudioMesh> sm;
        std::vector<Buf>        cmds;
        std::vector<int>        normOffsets;
        float                   radius = 0;

        for (int mi : groups[gi])
        {
            const OutMesh& m = meshes[mi];
            int voff = (int)verts.size() / 3, noff = (int)norms.size() / 3;

            StudioMesh s = {};
            s.numtris = (int)m.tris.size() / 3;
            s.skinref = m.skinref;
            s.numnorms = (int)m.norms.size() / 3;
            sm.push_back(s);
            normOffsets.push_back(noff);

            Buf c;
            for (size_t t = 0; t + 2 < m.tris.size(); t += 3)
            {
                c.Put((short)3);
                for (int k = 0; k < 3; k++)
                {
                    StudioTriVert tv = m.tris[t + k];
                    tv.vertindex = (short)(tv.vertindex + voff);
                    tv.normindex = (short)(tv.normindex + noff);
                    c.Put(tv);
                }
            }
            c.Put((short)0);
            cmds.push_back(std::move(c));

            verts.insert(verts.end(), m.verts.begin(), m.verts.end());
            norms.insert(norms.end(), m.norms.begin(), m.norms.end());
            vbone.insert(vbone.end(), m.vbone.begin(), m.vbone.end());
            nbone.insert(nbone.end(), m.nbone.begin(), m.nbone.end());
        }

        for (size_t i = 0; i + 2 < verts.size(); i += 3)
        {
            const Rigid& bb = bones[vbone[i / 3]].bind;
            V3 w = M3MulV(bb.R, { verts[i], verts[i + 1], verts[i + 2] }) + bb.t;
            radius = std::max(radius, Len(w));
        }

        size_t nv = verts.size() / 3, nn = norms.size() / 3;

        size_t viOff = b.Align(); b.PutRaw(vbone.data(), nv);
        size_t vOff = b.Align(); b.PutRaw(verts.data(), nv * 12);
        size_t niOff = b.Align(); b.PutRaw(nbone.data(), nn);
        size_t nOff = b.Align(); b.PutRaw(norms.data(), nn * 12);

        for (size_t i = 0; i < sm.size(); i++)
        {
            sm[i].triindex = (int)b.Align();
            b.PutRaw(cmds[i].d.data(), cmds[i].d.size());
            sm[i].normindex = 0;
        }

        size_t meshOff = b.Align();
        for (auto& s : sm) b.Put(s);

        StudioModel m = {};
        snprintf(m.name, sizeof(m.name), "sub%zu", gi);
        m.boundingradius = radius;
        m.nummesh = (int)sm.size();
        m.meshindex = (int)meshOff;
        m.numverts = (int)nv;
        m.vertinfoindex = (int)viOff;
        m.vertindex = (int)vOff;
        m.numnorms = (int)nn;
        m.norminfoindex = (int)niOff;
        m.normindex = (int)nOff;
        size_t modelOff = b.Align();
        b.Put(m);

        StudioBodypart bp = {};
        snprintf(bp.name, sizeof(bp.name), "body%zu", gi);
        bp.nummodels = 1;
        bp.base = 1;
        bp.modelindex = (int)modelOff;
        bodyparts.push_back(bp);
    }

    size_t bpOff = b.Align();
    for (auto& bp : bodyparts) b.Put(bp);
    b.Align();

    size_t texDataOff = b.Align();
    for (size_t i = 0; i < texs.size(); i++)
    {
        int idx = (int)b.Align();
        b.PutRaw(texs[i].pixels.data(), texs[i].pixels.size());
        b.PutRaw(texs[i].pal, 768);
        memcpy(&b.d[texOff + i * sizeof(StudioTexture) + offsetof(StudioTexture, index)], &idx, sizeof(int));
    }
    b.Align();

    std::string base = out;
    size_t slash = base.find_last_of("/\\");
    if (slash != std::string::npos) base.erase(0, slash + 1);

    StudioHdr h = {};
    memcpy(&h.id, "IDST", 4);
    h.version = 10;
    strncpy(h.name, base.c_str(), sizeof(h.name) - 1);
    h.length = (int)b.d.size();
    h.eyeposition[2] = gmax[2];
    memcpy(h.min, gmin, sizeof(gmin));
    memcpy(h.max, gmax, sizeof(gmax));
    memcpy(h.bbmin, gmin, sizeof(gmin));
    memcpy(h.bbmax, gmax, sizeof(gmax));
    h.numbones = nb;                  h.boneindex = (int)boneOff;
    h.numhitboxes = (int)hboxes.size();  h.hitboxindex = (int)hbOff;
    h.numseq = (int)seqs.size();    h.seqindex = (int)seqOff;
    h.numseqgroups = 1;                   h.seqgroupindex = (int)sgOff;
    h.numtextures = (int)texs.size();
    h.textureindex = (int)texOff;
    h.texturedataindex = (int)texDataOff;
    h.numskinref = (int)texs.size();
    h.numskinfamilies = 1;
    h.skinindex = (int)skinOff;
    h.numbodyparts = (int)bodyparts.size();
    h.bodypartindex = (int)bpOff;
    memcpy(b.d.data(), &h, sizeof(h));

    FILE* f = fopen(out.c_str(), "wb");
    if (!f)
    {
        printf("Error: could not write '%s'.\n", out.c_str());
        cg::cgltf_free(data);
        return 5;
    }
    fwrite(b.d.data(), 1, b.d.size(), f);
    fclose(f);

    int totalTris = 0, totalVerts = 0;
    for (auto& m : meshes) { totalTris += (int)m.tris.size() / 3; totalVerts += (int)m.verts.size() / 3; }
    printf("OK: %s (%zu bytes)\n", out.c_str(), b.d.size());
    printf("    %d bones, %zu hitboxes, %zu sequences (%d frames), %zu textures, %zu submodels\n",
        nb, hboxes.size(), seqs.size(), totalFrames, texs.size(), groups.size());
    printf("    %d vertices, %d triangles\n", totalVerts, totalTris);
    printf("    size (GoldSrc units): %.1f x %.1f x %.1f\n", gmax[0] - gmin[0], gmax[1] - gmin[1], gmax[2] - gmin[2]);
    {
        float big = std::max(gmax[0] - gmin[0], std::max(gmax[1] - gmin[1], gmax[2] - gmin[2]));
        if (big > 1024.0f || big < 4.0f)
            printf("    Hint: the final size looks odd for GoldSrc (a player is ~72 units tall).\n"
                "          Use --height 72 (final height) or --scale N to adjust.\n");
    }
    for (const SeqOut& s : seqs)
        printf("    seq '%s': %d frames at %.1f fps%s\n", s.name.c_str(), s.frames, s.fps, (s.flags & STUDIO_LOOPING) ? " (loop)" : "");

    cg::cgltf_free(data);
    return 0;
}

int main(int argc, char** argv)
{
    Opts o;
    bool interactive = false;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--scale") == 0 && i + 1 < argc)        o.scale = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc)  o.height = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc)     o.fps = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--no-flip") == 0)                 o.flip = false;
        else if (o.in.empty())                                      o.in = argv[i];
        else if (o.out.empty())                                     o.out = argv[i];
    }
    if (o.fps < 1) o.fps = 1;

    if (o.in.empty())
    {
        interactive = true;
        printf("GLB file: ");
        std::getline(std::cin, o.in);
        while (!o.in.empty() && (o.in.back() == ' ' || o.in.back() == '"' || o.in.back() == '\r')) o.in.pop_back();
        while (!o.in.empty() && (o.in.front() == ' ' || o.in.front() == '"')) o.in.erase(o.in.begin());
    }

    int rc = 1;
    if (o.in.empty())
        printf("No file given.\n");
    else
    {
        float scale = o.scale;
        rc = 0;
        if (o.height > 0)
        {
            float h1 = 0;
            rc = Convert(o, 1.0f, &h1);
            if (rc == 0)
            {
                if (h1 > 1e-6f) scale = o.height / h1;
                else printf("Warning: could not measure the height; using --scale %.2f.\n", scale);
            }
        }
        if (rc == 0) rc = Convert(o, scale, nullptr);
    }

    if (interactive)
    {
        printf("\nPress Enter to exit...");
        std::string dummy;
        std::getline(std::cin, dummy);
    }
    return rc;
}