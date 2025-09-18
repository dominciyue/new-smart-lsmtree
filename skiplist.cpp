#include "skiplist.h"
#include <cstdlib>
#include <ctime>
#include <random>
#include <iostream>

// 生成随机数，用于决定节点的层数
double skiplist::my_rand() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0, 1);
    return dis(gen);
}

// 随机生成节点的层数
int skiplist::randLevel() {
    int level = 1;
    while (my_rand() < p && level < MAX_LEVEL) {
        level++;
    }
    return level;
}

// 插入键值对
void skiplist::insert(uint64_t key, const std::string &str) {
    slnode *update[MAX_LEVEL];
    slnode *cur = head;
    
    // 查找插入位置，并记录每层的前驱节点
    for (int i = curMaxL - 1; i >= 0; --i) {
        while (cur->nxt[i]->key < key && cur->nxt[i]->type != TAIL) {
            cur = cur->nxt[i];
        }
        update[i] = cur;
    }
    
    cur = cur->nxt[0];
    
    // 如果键已存在，更新值
    if (cur->key == key && cur->type != TAIL) {
        uint32_t oldLen = cur->val.length();
        cur->val = str;
        bytes = bytes - oldLen + str.length();
        return;
    }
    
    // 生成随机层数
    int level = randLevel();
    
    // 如果新层数大于当前最大层数，更新最大层数
    if (level > curMaxL) {
        for (int i = curMaxL; i < level; ++i) {
            update[i] = head;
        }
        curMaxL = level;
    }
    
    // 创建新节点
    slnode *newNode = new slnode(key, str, NORMAL);
    
    // 更新指针
    for (int i = 0; i < level; ++i) {
        newNode->nxt[i] = update[i]->nxt[i];
        update[i]->nxt[i] = newNode;
    }
    
    // 更新节点数和字节数
    s++;
    bytes += 12 + str.length(); // 12 = 8(key) + 4(offset)
}

// 查询键对应的值
std::string skiplist::search(uint64_t key) {
    slnode *cur = head;

    for (int i = curMaxL - 1; i >= 0; --i) {
        while (cur->nxt[i]->key < key && cur->nxt[i]->type != TAIL) {
            cur = cur->nxt[i];
        }
    }

    cur = cur->nxt[0];

    if (cur->key == key && cur->type != TAIL) {
        return cur->val;
    }

    return "";
}

// 删除键值对
bool skiplist::del(uint64_t key, uint32_t len) {
    slnode *update[MAX_LEVEL];
    slnode *cur = head;
    
    // 查找删除位置，并记录每层的前驱节点
    for (int i = curMaxL - 1; i >= 0; --i) {
        while (cur->nxt[i]->key < key && cur->nxt[i]->type != TAIL) {
            cur = cur->nxt[i];
        }
        update[i] = cur;
    }
    
    cur = cur->nxt[0];
    
    // 如果键不存在，返回false
    if (cur->key != key || cur->type == TAIL) {
        return false;
    }
    
    // 更新指针，删除节点
    for (int i = 0; i < curMaxL; ++i) {
        if (update[i]->nxt[i] != cur) {
            break;
        }
        update[i]->nxt[i] = cur->nxt[i];
    }
    
    // 更新字节数
    bytes -= 12 + cur->val.length();
    
    // 释放内存
    delete cur;
    
    // 更新最大层数
    while (curMaxL > 1 && head->nxt[curMaxL - 1] == tail) {
        curMaxL--;
    }
    
    // 更新节点数
    s--;
    
    return true;
}

// 范围查询
void skiplist::scan(uint64_t key1, uint64_t key2, std::vector<std::pair<uint64_t, std::string>> &list) {
    slnode *cur = lowerBound(key1);
    
    // 遍历范围内的所有节点
    while (cur->type != TAIL && cur->key <= key2) {
        list.emplace_back(cur->key, cur->val);
        cur = cur->nxt[0];
    }
}

// 查找大于等于key的第一个节点
slnode *skiplist::lowerBound(uint64_t key) {
    slnode *cur = head;
    
    // 从最高层开始查找
    for (int i = curMaxL - 1; i >= 0; --i) {
        while (cur->nxt[i]->key < key && cur->nxt[i]->type != TAIL) {
            cur = cur->nxt[i];
        }
    }
    
    return cur->nxt[0];
}

// 重置跳表
void skiplist::reset() {
    // 删除所有节点（除了头尾节点）
    slnode *cur = head->nxt[0];
    while (cur != tail) {
        slnode *next = cur->nxt[0];
        delete cur;
        cur = next;
    }
    
    // 重置指针
    for (int i = 0; i < MAX_LEVEL; ++i) {
        head->nxt[i] = tail;
    }
    
    // 重置状态
    s = 1;
    bytes = 0;
    curMaxL = 1;
}

// 获取字节数
uint32_t skiplist::getBytes() {
    return bytes;
}

// --- 新增 getCnt 实现 ---
uint64_t skiplist::getCnt() const {
    return s; // 返回私有成员 s
}
// --- 结束新增 ---
